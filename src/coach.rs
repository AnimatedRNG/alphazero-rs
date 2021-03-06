use crossbeam::{channel, scope};
use log::{info, warn};
use ndarray::{ArcArray, Array, Axis, Ix1, IxDyn};
use pbr::ProgressBar;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::thread;

use crate::arena::{play_games, GameResult};
use crate::async_mcts::AsyncMcts;
use crate::game::Game;
use crate::nnet::*;

pub struct Coach {
    history: VecDeque<VecDeque<TrainingSample>>,
    mcts_reserve_size: usize,
    update_threshold: f32,
    temp_threshold: usize,
    max_history_length: usize,
    max_queue_length: usize,
    inference_batch_size: usize,
    num_episode_threads: usize,
    num_arena_games: usize,
    num_iters: usize,
    num_eps: usize,
    num_sims: usize,
    num_sim_threads: usize,
    max_depth: usize,
    cpuct: i32,
}

impl Coach {
    #[allow(clippy::too_many_arguments)]
    pub fn setup<P: AsRef<Path>>(
        checkpoint_directory: P,
        mcts_reserve_size: usize,
        update_threshold: f32,
        temp_threshold: usize,
        max_history_length: usize,
        max_queue_length: usize,
        inference_batch_size: usize,
        num_episode_threads: usize,
        num_arena_games: usize,
        num_iters: usize,
        num_eps: usize,
        num_sims: usize,
        num_sim_threads: usize,
        max_depth: usize,
        cpuct: i32,
    ) -> Coach {
        let checkpoint_dir_paths = fs::read_dir(&checkpoint_directory);

        let history: VecDeque<VecDeque<TrainingSample>> = match checkpoint_dir_paths {
            Ok(checkpoint_dir_paths) => {
                let most_recent_checkpoint = checkpoint_dir_paths
                    .max_by_key(|path| {
                        path.as_ref()
                            .unwrap()
                            .path()
                            .file_stem()
                            .unwrap()
                            .to_str()
                            .unwrap()
                            .parse::<usize>()
                            .unwrap()
                    })
                    .unwrap()
                    .unwrap()
                    .path();
                let checkpoint = fs::read(most_recent_checkpoint).unwrap();
                bincode::deserialize(&checkpoint).unwrap()
            }
            Err(_) => {
                fs::create_dir(checkpoint_directory).unwrap();
                VecDeque::new()
            }
        };

        assert!(num_sims % inference_batch_size == 0);

        Coach {
            history,
            mcts_reserve_size,
            update_threshold,
            temp_threshold,
            max_history_length,
            max_queue_length,
            inference_batch_size,
            num_episode_threads,
            num_arena_games,
            num_iters,
            num_eps,
            num_sims,
            num_sim_threads,
            max_depth,
            cpuct,
        }
    }

    pub fn execute_episode<G: Game>(
        &self,
        mcts: AsyncMcts<G>,
        episode_id: usize,
        rng: &mut SmallRng,
    ) -> VecDeque<TrainingSample> {
        let mut train_examples: Vec<(BoardFeatures, i8, Policy)> = Vec::new();

        let mut board = G::get_init_board();

        let mut cur_player: i8 = 1;

        let mut episode_step: usize = 0;

        loop {
            episode_step += 1;
            let canonical_board = board.get_canonical_form(cur_player);

            let temp = if episode_step < self.temp_threshold {
                1.0
            } else {
                0.0
            };

            let pi = mcts.get_action_prob(&canonical_board, temp, episode_id, rng);

            let sym = canonical_board.get_symmetries(pi.view());

            train_examples.extend(
                sym.into_iter()
                    .map(|(b, p)| (b.to_features(), cur_player, p)),
            );

            let pi_enum: Vec<_> = pi.iter().enumerate().collect();
            let action = pi_enum.choose_weighted(rng, |(_, &p)| p).unwrap().0 as u8;

            let next_state = board.get_next_state(cur_player, action);
            board = next_state.0;
            cur_player = next_state.1;

            let r = board.get_game_ended(cur_player);

            if r != 0.0 {
                return train_examples
                    .into_iter()
                    .map(|(b_f, player, p)| TrainingSample {
                        board: b_f,
                        pi: p,
                        v: if player == cur_player { 1.0 } else { -1.0 },
                    })
                    .collect();
            }
        }
    }

    pub fn save_train_examples<P: AsRef<Path>>(&self, iteration: usize, checkpoint: P) {
        let filename = checkpoint
            .as_ref()
            .join(Path::new(&format!("/{}.examples", iteration)));

        let encoded = bincode::serialize(&self.history).unwrap();
        fs::write(&filename, encoded)
            .unwrap_or_else(|_| panic!("unable to write iteration {}", iteration));
    }

    pub fn learn<G: Game, N: NNet, P: AsRef<Path> + Send>(
        &mut self,
        checkpoint: P,
        skip_first_play: bool,
        verbose: bool,
        rng: &mut SmallRng,
    ) {
        scope(|scope| {
            let mut model_id = 0;

            // set up inference thread
            let (tx_give, rx_give) = channel::bounded(0);
            let (tx_data, rx_data) = channel::unbounded();
            let (tx_train, rx_train) = channel::bounded(0);

            let feature_shape = G::get_feature_shape();

            let inference_batch_size = self.inference_batch_size;

            let checkpoint_pbuf = checkpoint.as_ref().to_path_buf();

            // persistent inference thread
            let inference_thread = scope.spawn(move |_| {
                AsyncMcts::<G>::inference_thread::<N>(
                    checkpoint_pbuf,
                    rx_data,
                    rx_give,
                    rx_train,
                    inference_batch_size,
                    feature_shape,
                )
            });

            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(self.num_episode_threads)
                .build()
                .unwrap();

            for iteration in 0..self.num_iters {
                let pb = if verbose {
                    let mut pb = ProgressBar::new(self.num_eps as u64);
                    pb.format("╢▌▌░╟");
                    pb.message(&format!("Iteration {}", iteration));
                    Some(pb)
                } else {
                    None
                };
                let pb_send = pb.is_some();

                let mut iteration_train_examples: VecDeque<TrainingSample> = VecDeque::new();

                let (pb_tx, pb_rx) = channel::unbounded();

                // progress bar in separate thread
                let pb_thread = if pb_send {
                    let mut pb = pb.unwrap();
                    Some(thread::spawn(move || loop {
                        match pb_rx.recv() {
                            Ok(_) => {
                                pb.inc();
                            }
                            Err(_) => {
                                pb.finish();
                                break;
                            }
                        }
                    }))
                } else {
                    None
                };

                if !skip_first_play || iteration > 0 {
                    pool.install(|| {
                        iteration_train_examples.extend(
                            (0..self.num_eps)
                                .into_par_iter()
                                .map(|episode_id| {
                                    let mcts = AsyncMcts::<G>::default(
                                        self.mcts_reserve_size,
                                        self.num_sims,
                                        self.num_sim_threads,
                                        self.max_depth,
                                        model_id,
                                        self.cpuct,
                                        tx_give.clone(),
                                        tx_data.clone(),
                                    );

                                    // cloned rng state
                                    let mut rng = rng.clone();

                                    let result = self.execute_episode(mcts, episode_id, &mut rng);

                                    // advance episode progress bar
                                    if pb_send {
                                        pb_tx.send(()).unwrap()
                                    }

                                    result
                                })
                                .flatten()
                                .collect::<VecDeque<_>>(),
                        )
                    });

                    // only keep max_queue_length samples
                    while iteration_train_examples.len() > self.max_queue_length {
                        iteration_train_examples.pop_front();
                    }

                    if let Some(pb_thread) = pb_thread {
                        pb_thread.join().unwrap();
                    }
                }

                self.history.push_back(iteration_train_examples);

                if self.history.len() > self.max_history_length {
                    warn!("History is too long, removing last entry");
                    self.history.pop_front();
                }

                info!("Saving train examples...");
                self.save_train_examples(iteration, &checkpoint);
                info!("Saved!");

                info!("Shuffling train examples...");
                let mut all_train_examples: Vec<_> = self.history.iter().flatten().collect();
                all_train_examples.shuffle(rng);
                info!("Shuffled!");

                info!("Converting from AOS to SOA...");
                let num_samples = all_train_examples.len();
                let mut batched_board_features_shape = G::get_feature_shape();
                batched_board_features_shape.insert(0, num_samples);

                assert!(num_samples > 0);

                let policy_size = all_train_examples[0].pi.shape()[0];
                let mut samples: SOATrainingSamples = (
                    ArcArray::zeros(IxDyn(&batched_board_features_shape)),
                    ArcArray::zeros([num_samples, policy_size]),
                    ArcArray::zeros([num_samples]),
                );

                // convert AOS to SOA
                for i in 0..num_samples {
                    samples
                        .0
                        .index_axis_mut(Axis(0), i)
                        .assign(&all_train_examples[i].board);
                    samples
                        .1
                        .index_axis_mut(Axis(0), i)
                        .assign(&all_train_examples[i].pi);
                    samples.2[i] = all_train_examples[i].v;
                }
                info!("Converted!");

                info!("Training...");
                tx_train.send((samples, model_id, model_id + 1)).unwrap();
                info!("Trained!");

                // Old MCTS
                let pmcts = AsyncMcts::<G>::default(
                    self.mcts_reserve_size,
                    self.num_sims,
                    self.num_sim_threads,
                    self.max_depth,
                    model_id,
                    self.cpuct,
                    tx_give.clone(),
                    tx_data.clone(),
                );

                // New (trained) MCTS
                let nmcts = AsyncMcts::<G>::default(
                    self.mcts_reserve_size,
                    self.num_sims,
                    self.num_sim_threads,
                    self.max_depth,
                    model_id + 1,
                    self.cpuct,
                    tx_give.clone(),
                    tx_data.clone(),
                );

                let argmax = |p: Array<f32, Ix1>| {
                    p.indexed_iter()
                        .max_by(|(_, av), (_, bv)| {
                            av.partial_cmp(bv).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap()
                        .0 as u8
                };

                let win_counter = play_games::<G>(
                    self.num_arena_games,
                    vec![
                        // temp = 0.0, episode_id = [0, 1] -- avoids conflicts with
                        // inference thread
                        &|s| argmax(nmcts.get_action_prob(s, 0.0, 0, &mut rng.clone())),
                        &|s| argmax(pmcts.get_action_prob(s, 0.0, 1, &mut rng.clone())),
                    ],
                    None,
                    false,
                );

                let nwins = win_counter[&GameResult::Win];
                let pwins = win_counter[&GameResult::Loss];
                let draws = win_counter[&GameResult::Draw];

                info!("NEW/PREV WINS : {} / {}; DRAWS : {}", nwins, pwins, draws);

                if pwins + nwins == 0
                    || nwins as f32 / ((pwins + nwins) as f32) < self.update_threshold
                {
                    info!("REJECTING NEW MODEL");
                } else {
                    info!("ACCEPTING NEW MODEL");
                    model_id += 1;
                }
            }

            inference_thread.join().unwrap();
        })
        .unwrap();
    }
}
