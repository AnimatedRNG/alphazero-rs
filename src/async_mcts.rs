use crossbeam::{channel, scope, select, Receiver, Sender};
use log::warn;
use ndarray::{Array, Axis, Ix1, Zip};
use rand::rngs::SmallRng;
use rand::seq::IteratorRandom;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::game::Game;
use crate::nnet::*;
use crate::node::{NodeState, NodeStore};

pub struct AsyncMcts<G: Game> {
    reserve_space: usize,
    nodes: NodeStore<G>,
    num_sims: usize,
    num_threads: usize,
    max_depth: usize,
    model_id: usize,
    cpuct: i32,
    tx_give: Sender<(usize, Sender<(ArcPolicy, f32)>)>,
    tx_data: Sender<(usize, usize, ArcBoardFeatures)>,
}

impl<G: Game> AsyncMcts<G> {
    pub fn default(
        reserve_space: usize,
        num_sims: usize,
        num_threads: usize,
        max_depth: usize,
        model_id: usize,
        cpuct: i32,
        tx_give: Sender<(usize, Sender<(ArcPolicy, Value)>)>,
        tx_data: Sender<(usize, usize, ArcBoardFeatures)>,
    ) -> Self {
        AsyncMcts {
            reserve_space,
            nodes: NodeStore::<G>::new(reserve_space),
            num_sims,
            num_threads,
            max_depth,
            model_id,
            tx_give,
            tx_data,
            cpuct,
        }
    }

    pub fn from_state(
        s: G,
        reserve_space: usize,
        num_sims: usize,
        num_threads: usize,
        max_depth: usize,
        model_id: usize,
        cpuct: i32,
        tx_give: Sender<(usize, Sender<(ArcPolicy, Value)>)>,
        tx_data: Sender<(usize, usize, ArcBoardFeatures)>,
    ) -> Self {
        AsyncMcts {
            reserve_space,
            nodes: NodeStore::<G>::from_root(reserve_space, s),
            num_sims,
            num_threads,
            max_depth,
            model_id,
            tx_give,
            tx_data,
            cpuct,
        }
    }

    pub fn get_action_prob(
        &self,
        s: &G,
        temp: f32,
        episode_id: usize,
        thread_rng: &mut SmallRng,
    ) -> Policy {
        let root_node_idx = self.nodes.lookup_state_id(s).unwrap();
        self.search(root_node_idx, episode_id);

        let root_node = self.nodes.get(root_node_idx).unwrap();
        let num_actions = root_node.mu.p.as_ref().unwrap().len();

        let mut counts = Array::zeros(num_actions);
        for child_idx in &root_node.children {
            let child = self.nodes.get(*child_idx).unwrap();
            let a = child.a;
            let n = child.get_n();

            counts[a as usize] = n;
        }

        let mut probs = Array::zeros(num_actions);
        if temp == 0.0 {
            let max_val = counts.iter().max().unwrap();
            let best_a = counts
                .iter()
                .enumerate()
                .filter(|(_, &c)| c == *max_val)
                .map(|(i, _)| i)
                .choose(thread_rng)
                .unwrap();
            probs[best_a] = 1.0;
            probs
        } else {
            let counts = counts.mapv(|c: u16| f32::powf(c as f32, 1.0 / temp));
            let counts_sum = counts.sum();
            probs.mapv_inplace(|x| x / counts_sum);

            probs
        }
    }

    pub fn inference_thread<N: NNet>(
        checkpoint_dir: PathBuf,
        rx: Receiver<(usize, usize, ArcBoardFeatures)>,
        rx_send: Receiver<(usize, Sender<(ArcPolicy, Value)>)>,
        rx_train: Receiver<(SOATrainingSamples, usize, usize)>,
        batch_size: usize,
        feature_shape: Vec<usize>,
    ) {
        let mut nnet: N = N::new(&checkpoint_dir);

        let mut tx_ret: HashMap<usize, Sender<(ArcPolicy, Value)>> = HashMap::new();

        let mut board_shape = feature_shape.clone();
        board_shape.insert(0, batch_size);

        let mut inference_batch = BatchedBoardFeatures::zeros(board_shape);
        let mut batch_ids: Vec<usize> = Vec::new(); // batch id -> thread id

        loop {
            select! {
                recv(rx) -> msg => {
                    match msg {
                        Err(_) => {
                            break;
                        },
                        Ok((i, model_id, board)) => {
                            // write batch into the buffer
                            batch_ids.push(i);
                            let batch_id = batch_ids.len();
                            let mut batch_subview = inference_batch.index_axis_mut(Axis(0), batch_id);
                            batch_subview.assign(&board.view());

                            // when we need to respond to everyone
                            if batch_ids.len() == batch_size {
                                let (pis, vs): (BatchedPolicy, BatchedValue) = nnet.predict(inference_batch.view(), model_id);

                                for (batch_id, thread_id) in batch_ids.iter().enumerate() {
                                    let pi = pis.index_axis(Axis(0), batch_id).into_owned().into_shared();
                                    let v = vs[batch_id];
                                    tx_ret.get(&thread_id).unwrap().send((pi, v)).unwrap();
                                }

                                // reset the batch ids vector
                                // no need to actually clear out inference_batch --
                                // it'll get overwritten soon
                                batch_ids.clear();
                            }
                        }
                    }
                },
                recv(rx_send) -> msg => {
                    match msg {
                        Err(_) => {
                            break;
                        },
                        Ok((i, tx)) => {
                            tx_ret.insert(i, tx);
                        }
                    }
                },
                recv(rx_train) -> msg => {
                    match msg {
                        Err(_) => {
                            break;
                        },
                        Ok((train_data, old_model_id, model_id)) => {
                            nnet.train(train_data, old_model_id, model_id);
                        }
                    }
                }
            }
        }
    }

    pub fn search(&self, root_idx: usize, episode_id: usize) {
        assert!(self.num_sims % self.num_threads == 0);

        let sim_id = AtomicUsize::new(0);

        scope(|scope| {
            (0..self.num_threads).for_each(|thread_id| {
                let thread_id = thread_id + self.num_threads * episode_id;

                // clone it?
                let tx_give = self.tx_give.clone();
                let tx_data = self.tx_data.clone();
                let s = &self;
                let sim_id = &sim_id;

                scope.spawn(move |_| {
                    let (tx, rx) = channel::bounded(1);
                    tx_give.send((thread_id, tx)).unwrap();

                    while sim_id.fetch_add(1, Ordering::SeqCst) < self.num_sims {
                        s.search_iteration(root_idx, thread_id, &tx_data, &rx);
                    }
                });
            });
        })
        .unwrap();
    }

    fn search_iteration(
        &self,
        root_idx: usize,
        thread_id: usize,
        nnet_tx: &Sender<(usize, usize, ArcBoardFeatures)>,
        nnet_rx: &Receiver<(ArcPolicy, Value)>,
    ) {
        let mut current_head_id = root_idx;
        let mut current_head_state = NodeState::Exists(true);

        let mut node_path = Vec::new();
        node_path.reserve(self.max_depth);

        let mut p_s_id = 0;

        let mut depth = 0;

        let v = loop {
            match current_head_state {
                NodeState::Exists(_) => {
                    let current_head = self.nodes.get(current_head_id).unwrap();

                    if depth > self.max_depth {
                        // heuristic handling
                        break current_head.mu.s.as_ref().unwrap().eval_heuristic();
                    }

                    let e = current_head.e.load();
                    if e != 0.0 {
                        break e;
                    }

                    current_head.visit();

                    let mut first_iteration = true;
                    loop {
                        let best_child_id =
                            self.nodes
                                .best_child(current_head_id, self.cpuct, !first_iteration);
                        first_iteration = false;

                        match self.nodes.state(best_child_id).unwrap() {
                            NodeState::PlaceHolder => {
                                if self.nodes.lock(best_child_id) {
                                    // if we descended onto a leaf, keep track of our
                                    // parent id for just a sec
                                    p_s_id = current_head_id;
                                    break;
                                }
                            }
                            NodeState::Exists(best_child_state) => {
                                node_path.push(current_head_id);
                                current_head_id = best_child_id;
                                current_head_state = NodeState::Exists(best_child_state);
                                depth += 1;
                            }
                            _ => continue,
                        };
                    }
                }
                NodeState::Locked => {
                    let node_p = self.nodes.get(p_s_id).unwrap();
                    let node_p_s = node_p.mu.s.as_ref().unwrap();

                    // we managed to grab a leaf node
                    let (next_state, next_player) = node_p_s.get_next_state(1, node_p.a);

                    // s is the state vector
                    let s = next_state.get_canonical_form(next_player);

                    match self.nodes.upgrade(current_head_id, s) {
                        None => {
                            panic!("Upgraded invalid node!");
                        }
                        Some(false) => {
                            // upgraded into duplicate node, just
                            // continue from the resolved node

                            current_head_id = self.nodes.resolve(current_head_id).unwrap();
                            current_head_state = NodeState::Exists(true);
                        }
                        Some(true) => {
                            // node was upgraded and children were populated.
                            // it remains locked. submit a prediction.
                            let current_head = self.nodes.get(current_head_id).unwrap();
                            let current_head_s = current_head.mu.s.as_ref().unwrap();
                            let features: BoardFeatures = current_head_s.to_features();

                            // techically we haven't "visited" this node yet,
                            // in the previous clause
                            current_head.visit();

                            nnet_tx
                                .send((thread_id, self.model_id, features.into_shared()))
                                .unwrap();

                            let (pi, v) = nnet_rx.recv().unwrap();

                            let mut pi: Policy = pi.into_owned();

                            let valids = current_head.mu.v.as_ref().unwrap();

                            // mask invalid moves
                            Zip::from(&mut pi).and(valids).apply(|x, &valid| {
                                if valid == 0 {
                                    *x = 0.0;
                                }
                            });

                            let sum_ps = pi.sum();

                            if sum_ps > 0.0 {
                                pi /= sum_ps; // normalize
                            } else {
                                // no valid moves, pick randomly
                                // hopefully shouldn't happen often?

                                // kind of gross, just add valids

                                warn!("Masked all valid moves -- picking randomly");

                                Zip::from(&mut pi).and(valids).apply(|x, &valid| {
                                    *x += valid as f32;
                                });

                                pi /= pi.sum();
                            }

                            // update pi
                            self.nodes.set_policy(current_head_id, pi);

                            // finally unlock this node
                            self.nodes.unlock(current_head_id);

                            break -v;
                        }
                    };
                }
                NodeState::PlaceHolder => panic!("Node was not upgraded somehow?"),
            };
        };

        loop {
            let current_head = self.nodes.get(current_head_id).unwrap();
            current_head.unvisit(v);

            if current_head_id == root_idx {
                break;
            }

            current_head_id = node_path.pop().unwrap();
        }
    }
}
