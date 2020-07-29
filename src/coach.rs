use crossbeam::{channel, scope};
use pbr::ProgressBar;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::thread;

use crate::async_mcts::AsyncMcts;
use crate::game::{Game, F};
use crate::nnet::NNet;
use crate::nnet::{BoardFeatures, BoardFeaturesView, Policy, TrainingSample, Value};

pub struct Coach {
    history: Vec<VecDeque<TrainingSample>>,
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
}

impl Coach {
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

            let pi = mcts
                .get_action_prob(&canonical_board, temp, episode_id, rng);

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

    pub fn learn<G: Game, N: NNet>(&mut self, verbose: bool, rng: &mut SmallRng) {
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

                let mut iteration_train_examples = VecDeque::new();

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

                            self.execute_episode(mcts, episode_id, &mut rng);

                            // advance episode progress bar
                            if pb_send {
                                pb_tx.send(()).unwrap()
                            }
                        })
                        .collect::<VecDeque<_>>(),
                );

                pb_thread.map(|pb_thread| pb_thread.join().unwrap());
            }

            inference_thread.join().unwrap();
        })
        .unwrap();
    }
}
