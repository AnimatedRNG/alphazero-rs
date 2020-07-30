use crossbeam::{channel, scope};
use pbr::ProgressBar;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::thread;

use crate::async_mcts::AsyncMcts;
use crate::game::Game;
use crate::nnet::NNet;
use crate::nnet::{BoardFeatures, Policy, TrainingSample};

pub struct Coach {
    history: Vec<VecDeque<TrainingSample>>,
    _update_threshold: f32,
    temp_threshold: usize,
    max_history_length: usize,
    num_episode_threads: usize,
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
        update_threshold: f32,
        temp_threshold: usize,
        max_history_length: usize,
        num_episode_threads: usize,
        num_iters: usize,
        num_eps: usize,
        num_sims: usize,
        num_sim_threads: usize,
        max_depth: usize,
        cpuct: i32,
    ) -> Coach {
        let checkpoint_dir_paths = fs::read_dir(&checkpoint_directory);

        let history: Vec<VecDeque<TrainingSample>> = match checkpoint_dir_paths {
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
                Vec::new()
            }
        };

        Coach {
            history,
            _update_threshold: update_threshold,
            temp_threshold,
            max_history_length,
            num_episode_threads,
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

            if r != 0 {
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

        //let json_rep = serde_json::to_string(&examples).unwrap();
        //fs::write(&filename, json_rep).expect(&format!("unable to write iteration {}", iteration));

        let encoded = bincode::serialize(&self.history).unwrap();
        fs::write(&filename, encoded)
            .unwrap_or_else(|_| panic!("unable to write iteration {}", iteration));
    }

    pub fn learn<G: Game, N: NNet, P: AsRef<Path>>(
        &mut self,
        checkpoint: P,
        skip_first_play: bool,
        verbose: bool,
        rng: &mut SmallRng,
    ) {
        scope(|scope| {
            // set up inference thread
            let (tx_give, rx_give) = channel::bounded(0);
            let (tx_data, rx_data) = channel::unbounded();

            let feature_shape = G::get_feature_shape();

            let nnet = N::new();

            // persistent inference thread
            let inference_thread = scope.spawn(move |_| {
                AsyncMcts::<G>::inference_thread(rx_data, rx_give, nnet, feature_shape)
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
                                        self.num_sims,
                                        self.num_sim_threads,
                                        self.max_depth,
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

                    if let Some(pb_thread) = pb_thread {
                        pb_thread.join().unwrap();
                    }
                }

                self.history.push(iteration_train_examples);

                if self.history.len() > self.max_history_length {
                    self.history.pop();
                }

                self.save_train_examples(iteration, &checkpoint);
            }

            inference_thread.join().unwrap();
        })
        .unwrap();
    }
}
