use ccl::dhashmap::DHashMap;
use crossbeam::atomic::AtomicCell;
use ndarray::{Array, Ix1};
use std::cell::UnsafeCell;
use std::marker::PhantomPinned;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use crate::game::Game;
use std::hash::{Hash, Hasher};

const EPS: f32 = 1e-6;
const WIN_SCALE: f32 = 100.0f32;

#[derive(Debug)]
pub struct Node<G: Game> {
    pub win_counter: AtomicU64, // 0xWWWWWWWWNNNNVVVV
    pub win_scale: f32,
    pub a: u8,
    pub e: AtomicCell<f32>,
    pub mu: NodeMutableState<G>,
    pub children: Vec<usize>,
    _pin: PhantomPinned,
}

#[derive(Clone, Debug)]
pub struct NodeMutableState<G: Game> {
    pub p: Option<Array<f32, Ix1>>,
    pub v: Option<Array<u8, Ix1>>,
    pub s: Option<G>,
}

impl<G: Game> Node<G> {
    pub fn empty(win_scale: f32) -> Node<G> {
        Node {
            win_counter: AtomicU64::new(0x7FFFFFFF00000000),
            win_scale,
            a: 0,
            e: AtomicCell::new(0.0),
            mu: NodeMutableState {
                p: None,
                v: None,
                s: None,
            },
            children: Vec::new(),
            _pin: PhantomPinned,
        }
    }

    #[inline]
    pub fn compute_q(&self) -> f32 {
        let n = self.get_n();
        if n > 0 {
            (self.get_w() - self.get_vloss() as f32) / (n as f32)
        } else {
            0.0
        }
    }

    #[inline]
    pub fn get_w(&self) -> f32 {
        ((self.win_counter.load(Ordering::Acquire) >> 32) as i64 - 0x7FFFFFFF) as f32
            / self.win_scale
    }

    #[inline]
    pub fn get_n(&self) -> u16 {
        ((self.win_counter.load(Ordering::Acquire) & 0x00000000FFFF0000u64) >> 16) as u16
    }

    #[inline]
    pub fn get_vloss(&self) -> u16 {
        (self.win_counter.load(Ordering::Acquire) & 0x000000000000FFFFu64) as u16
    }

    #[inline]
    pub fn visit(&self) {
        self.win_counter
            .fetch_add(0x0000000000010001u64, Ordering::SeqCst);
    }

    #[inline]
    pub fn unvisit(&self, win_val: f32) {
        let incr = (self.win_scale * win_val).abs() as u32;
        let incr = if win_val < 0f32 {
            (incr as u64) << 32
        } else {
            ((0xFFFFFFFF - incr) as u64) << 32
        };
        self.win_counter
            .fetch_sub(0x0000000000000001u64 | incr, Ordering::SeqCst);
    }
}

impl<G: Game> Clone for Node<G> {
    fn clone(&self) -> Node<G> {
        Node {
            win_counter: AtomicU64::new(self.win_counter.load(Ordering::Acquire)),
            win_scale: self.win_scale,
            a: self.a,
            e: AtomicCell::new(self.e.load()),
            mu: self.mu.clone(),
            children: self.children.clone(),
            _pin: PhantomPinned,
        }
    }
}

impl<'a, G: Game> PartialEq for Node<G> {
    fn eq(&self, other: &Self) -> bool {
        if self.mu.s.is_none() && other.mu.s.is_none() {
            true
        } else if self.mu.s.is_some() && other.mu.s.is_some() {
            self.mu.s == other.mu.s
        } else {
            false
        }
    }
}

impl<G: Game> Eq for Node<G> {}

impl<G: Game> Hash for Node<G> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.mu.s.as_ref().unwrap().hash(hasher);
    }
}

type NodeLink<G> = (Node<G>, Option<usize>);
type NodeMutex<G> = (AtomicBool, UnsafeCell<Option<NodeLink<G>>>);

pub struct NodeStore<G: Game> {
    buf: Vec<NodeMutex<G>>,
    pub len: AtomicUsize,
    pub seen: DHashMap<G, usize>,
}

#[derive(PartialEq)]
pub enum NodeState {
    PlaceHolder,
    Locked,
    Exists(bool),
}

impl<G: Game> NodeStore<G> {
    pub fn empty(reserve_space: usize) -> Self {
        NodeStore {
            buf: (0..reserve_space)
                .map(|_| (AtomicBool::new(false), UnsafeCell::new(None)))
                .collect(),
            len: AtomicUsize::new(0),
            seen: DHashMap::default(),
        }
    }

    pub fn new(reserve_space: usize) -> Self {
        let s = G::get_init_board();
        let root = Node::empty(WIN_SCALE);

        let nodestore = NodeStore::empty(reserve_space);

        let root_idx = nodestore.push(root);
        nodestore.upgrade(root_idx, s);

        nodestore
    }

    pub fn from_root(reserve_space: usize, s: G) -> Self {
        let root = Node::empty(WIN_SCALE);

        let nodestore = NodeStore::empty(reserve_space);

        let root_idx = nodestore.push(root);
        nodestore.upgrade(root_idx, s);

        nodestore
    }

    pub fn resolve(&self, idx: usize) -> Option<usize> {
        let mut l = idx;
        let len = self.len.load(Ordering::Acquire);
        loop {
            if l >= len {
                return None;
            }

            match unsafe { &self.buf[l].1.get().as_ref().unwrap() } {
                Some((_, None)) => return Some(l),
                Some((_, Some(s))) => l = *s,
                None => return None,
            };
        }
    }

    pub fn get(&self, idx: usize) -> Option<Pin<&Node<G>>> {
        let l = self.resolve(idx);

        l.map(|l| unsafe {
            Pin::new_unchecked(&self.buf[l].1.get().as_ref().unwrap().as_ref().unwrap().0)
        })
    }

    pub fn lookup_state_id(&self, s: &G) -> Option<usize> {
        self.seen.get(s).map(|s| *s)
    }

    #[allow(unused)]
    pub fn lookup_state(&self, s: &G) -> Option<Pin<&Node<G>>> {
        self.seen.get(s).and_then(|idx| self.get(*idx))
    }

    pub fn set_policy(&self, idx: usize, policy: Array<f32, Ix1>) -> bool {
        let l = self.resolve(idx);

        if self.state(idx) != Some(NodeState::Locked) {
            false
        } else {
            let r = unsafe {
                &mut self.buf[l.unwrap()]
                    .1
                    .get()
                    .as_mut()
                    .unwrap()
                    .as_mut()
                    .unwrap()
                    .0
            };
            r.mu.p = Some(policy);

            true
        }
    }

    pub fn push(&self, node: Node<G>) -> usize {
        // increment the buffer pointer
        let idx = self.len.fetch_add(1, Ordering::SeqCst);
        assert!(idx < self.buf.len());

        let buf_idx = unsafe { self.buf[idx].1.get().as_mut().unwrap() };

        *buf_idx = Some((node, None));

        idx
    }

    pub fn state(&self, idx: usize) -> Option<NodeState> {
        if idx < self.len() {
            if self.buf[idx].0.load(Ordering::SeqCst) {
                Some(NodeState::Locked)
            } else {
                let nodelink = unsafe { self.buf[idx].1.get().as_ref().unwrap() };
                match nodelink.as_ref() {
                    None => None,
                    Some(nodelink) => {
                        if nodelink.1.is_none() {
                            if nodelink.0.mu.s.is_some() {
                                Some(NodeState::Exists(true))
                            } else {
                                Some(NodeState::PlaceHolder)
                            }
                        } else {
                            Some(NodeState::Exists(false))
                        }
                    }
                }
            }
        } else {
            None
        }
    }

    pub fn upgrade(&self, idx: usize, s: G) -> Option<bool> {
        if idx >= self.len() {
            None
        } else {
            let old_node = unsafe { self.buf[idx].1.get().as_mut().unwrap().as_mut().unwrap() };
            let link: &mut Option<usize> = &mut old_node.1;

            assert!(link.is_none() == true);

            let old_node = &mut old_node.0;
            let existing_idx = self.seen.get(&s);

            match existing_idx {
                Some(existing_idx) => {
                    *link = Some(*existing_idx);
                    self.unlock(idx);
                    Some(false)
                }
                None => {
                    let mu: &mut NodeMutableState<G> = &mut old_node.mu;
                    mu.s = Some(s.clone());
                    let game_ended = s.get_game_ended(1);
                    old_node.e.store((-game_ended) as f32);

                    if game_ended == 0 {
                        let valids = s.get_valid_moves(1);
                        let valid_actions: Vec<_> = valids
                            .indexed_iter()
                            .filter_map(
                                |(index, &item)| {
                                    if item != 0 {
                                        Some(index as u8)
                                    } else {
                                        None
                                    }
                                },
                            )
                            .collect();
                        old_node.children.reserve(valid_actions.len());
                        old_node.mu.v = Some(valids);

                        for a in valid_actions {
                            let mut child_node = Node::empty(WIN_SCALE);
                            child_node.a = a;
                            old_node.children.push(self.push(child_node));
                        }
                    }

                    self.seen.insert(s.clone(), idx);

                    Some(true)
                }
            }
        }
    }

    pub fn lock(&self, idx: usize) -> bool {
        self.buf[idx]
            .0
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok()
    }

    pub fn unlock(&self, idx: usize) {
        let unlocked = self.buf[idx]
            .0
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok();
        debug_assert!(unlocked);
    }

    pub fn best_child(&self, idx: usize, cpuct: i32, _filter: bool) -> usize {
        let node = self.get(idx).unwrap();

        let parent_n = node.get_n();
        let (best_child_idx, _) = node
            .children
            .iter()
            .map(|&child_idx| {
                let child = self.get(child_idx).unwrap();
                let u = child.compute_q()
                    + (cpuct as f32)
                        * node.mu.p.as_ref().unwrap()[child.a as usize]
                        * f32::sqrt(parent_n as f32 + EPS)
                        / (1 + child.get_n()) as f32;
                (child_idx, u)
            })
            .filter(|(child_idx, _)| {
                if _filter {
                    self.state(*child_idx) != Some(NodeState::Locked)
                } else {
                    true
                }
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        best_child_idx
    }

    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }
}

unsafe impl<G: Game> Sync for NodeStore<G> {}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    mod dummy_game;
    use super::*;
    use dummy_game::DummyGame;
    use rayon::prelude::*;
    use std::collections::HashMap;
    use std::ptr;

    fn similar(a: f32, b: f32, eps: f32) -> bool {
        f32::abs(a - b) < eps
    }

    #[test]
    fn test_win() {
        let node = Node::<DummyGame>::empty(10000.0f32);

        let initial_w = node.get_w();
        let initial_n = node.get_n();
        let initial_vloss = node.get_vloss();

        assert!(similar(initial_w, 0.0, 1e-3f32));
        assert!(initial_n == 0);
        assert!(initial_vloss == 0);

        node.visit();

        assert!(similar(node.get_w(), 0.0, 1e-3f32));
        assert!(node.get_n() == 1);
        assert!(node.get_vloss() == 1);

        node.unvisit(1.0);
        assert!(similar(node.get_w(), 1.0, 1e-3f32));
        assert!(node.get_n() == 1);
        assert!(node.get_vloss() == 0);
    }

    #[test]
    fn test_loss() {
        let node = Node::<DummyGame>::empty(10000.0f32);

        node.visit();
        node.unvisit(-1.0);
        assert!(similar(node.get_w(), -1.0, 1e-3f32));
        assert!(node.get_n() == 1);
        assert!(node.get_vloss() == 0);
    }

    #[test]
    fn test_winloss() {
        let node = Node::<DummyGame>::empty(10000.0f32);

        node.visit();
        node.unvisit(-1.0);
        node.visit();
        assert!(node.get_vloss() == 1);
        node.unvisit(1.0);
        assert!(similar(node.get_w(), 0.0, 1e-3f32));
        assert!(node.get_n() == 2);
        assert!(node.get_vloss() == 0);
    }

    #[test]
    fn test_is_lockfree() {
        assert!(AtomicCell::<usize>::is_lock_free(), true);
    }

    #[test]
    fn test_nodestore_empty() {
        let nodes = NodeStore::<DummyGame>::empty(2048);
        assert!(nodes.len() == 0);
    }

    #[test]
    fn test_nodestore_one() {
        let game = DummyGame::new(0);
        let nodes = NodeStore::<DummyGame>::empty(2048);

        let mut node = Node::<DummyGame>::empty(10000.0f32);
        node.mu = NodeMutableState {
            p: Some(Array::zeros(10)),
            v: Some(Array::zeros(10)),
            s: Some(game),
        };
        let idx = nodes.push(node.clone());

        assert!(unsafe { Some(Pin::into_inner_unchecked(nodes.get(idx).unwrap())) } == Some(&node));
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_nodestore_many() {
        let nodes = NodeStore::<DummyGame>::empty(8192);

        let mut idx = Vec::new();

        for i in 0..8192 {
            let node = Node::empty(10000.0f32);
            node.e.store(i as f32);
            idx.push(nodes.push(node));
        }
        assert!(nodes.len() == 8192);

        for i in 0..8192 {
            assert!(idx[i] as f32 == nodes.get(idx[i]).unwrap().e.load());
        }
    }

    #[test]
    fn test_nodestore_some_parallel() {
        let nodes = NodeStore::<DummyGame>::empty(1024);

        let idx: HashMap<usize, usize> = (0..1024)
            .into_par_iter()
            .map(|i| {
                let node = Node::empty(10000.0f32);
                node.e.store(i as f32);
                (i as usize, nodes.push(node))
            })
            .collect();
        assert!(nodes.len() == 1024);

        for i in 0..1024 {
            assert!(i as f32 == nodes.get(idx[&i]).unwrap().e.load());
        }
    }

    #[test]
    fn test_nodestore_many_parallel() {
        let nodes = NodeStore::<DummyGame>::empty(8192);

        let idx: HashMap<usize, usize> = (0..8192)
            .into_par_iter()
            .map(|i| {
                let node = Node::empty(10000.0f32);
                node.e.store(i as f32);
                (i as usize, nodes.push(node))
            })
            .collect();
        assert!(nodes.len() == 8192);

        for i in 0..8192 {
            assert!(i as f32 == nodes.get(idx[&i]).unwrap().e.load());
        }
    }

    #[test]
    fn test_nodestore_parallel_push_then_get() {
        let nodes = NodeStore::<DummyGame>::empty(8192);

        let idx: HashMap<usize, usize> = (0..8192)
            .into_par_iter()
            .map(|i| {
                let node = Node::empty(10000.0f32);
                node.e.store(i as f32);

                // not really a great test case
                if i > 32 && nodes.state((i - 32) as usize) == Some(NodeState::PlaceHolder) {
                    assert!(nodes.get((i - 32) as usize).is_some());
                }

                (i as usize, nodes.push(node))
            })
            .collect();
        assert!(nodes.len() == 8192);

        for i in 0..8192 {
            assert!(i as f32 == nodes.get(idx[&i]).unwrap().e.load());
        }
    }

    #[test]
    fn test_nodestore_upgrade_many_similar() {
        let nodes = NodeStore::<DummyGame>::empty(8192);

        let s = DummyGame::new(0);

        let idx: HashMap<usize, Option<usize>> = (0..8192)
            .into_iter()
            .map(|i| {
                let node = Node::empty(10000.0f32);
                let idx = nodes.push(node);
                assert!(nodes.lock(idx));
                let unique = nodes.upgrade(idx, s.clone()).unwrap();
                if unique {
                    assert!(nodes.state(idx) == Some(NodeState::Locked));
                    nodes.unlock(idx);
                }

                (i as usize, if unique { Some(idx) } else { None })
            })
            .collect();
        assert!(nodes.len() == 8192);

        assert!(
            idx.iter()
                .map(|(_, o)| if o.is_some() { 1 } else { 0 })
                .sum::<i32>()
                == 1
        );

        let root = idx
            .iter()
            .map(|(_, o)| o)
            .filter(|o| o.is_some())
            .map(|o| o.unwrap())
            .collect::<Vec<usize>>()[0];
        assert!(root == 0);
    }

    #[test]
    fn test_nodestore_upgrade() {
        let nodes = NodeStore::<DummyGame>::empty(2048);

        let node = Node::empty(10000.0f32);
        let idx = nodes.push(node.clone());

        let s = DummyGame::new(0);

        let const_ref = nodes.get(idx).unwrap();

        assert!(nodes.get(idx).unwrap().mu.s.is_none());

        // first insertion of s at location 0 succeeds
        assert!(nodes.lock(0));
        assert!(nodes.upgrade(0, s.clone()).unwrap());
        assert!(nodes.state(0) == Some(NodeState::Locked));

        let r_ref = nodes.get(idx).unwrap();

        // old const ref and new mutable reference should be
        // pointing to the same location in memory
        unsafe {
            assert!(ptr::eq(
                std::pin::Pin::<&Node<DummyGame>>::into_inner_unchecked(const_ref),
                std::pin::Pin::<&Node<DummyGame>>::into_inner_unchecked(r_ref)
            ));
        }

        nodes.unlock(0);

        assert!(nodes.state(0) == Some(NodeState::Exists(true)));

        assert!(nodes.get(idx).unwrap().mu.s.clone().unwrap() == s);
        assert!(nodes.seen.contains_key(&s));
        assert!(nodes.seen.len() == 1);

        // insertion of s at location 1 creates a link, already seen
        nodes.push(node);
        assert!(nodes.lock(1));
        assert!(!nodes.upgrade(1, s).unwrap());
        assert!(nodes.state(1) == Some(NodeState::Exists(false)));
    }

    #[test]
    fn test_nodestore_lock() {
        let nodes = NodeStore::<DummyGame>::empty(2048);

        let node = Node::empty(10000.0f32);
        let idx = nodes.push(node);
        assert!(nodes.state(idx) == Some(NodeState::PlaceHolder));

        let s = DummyGame::new(0);

        assert!(nodes.lock(idx));
        nodes.upgrade(idx, s).unwrap();

        // trying to lock a second time fails
        assert!(!nodes.lock(idx));
        assert!(nodes.state(idx) == Some(NodeState::Locked));

        nodes.unlock(idx);

        // now it is unlocked
        assert!(nodes.state(idx) == Some(NodeState::Exists(true)));
    }
}
