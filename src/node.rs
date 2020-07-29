use array_init::array_init;
use ccl::dhashmap::DHashMap;
use crossbeam::atomic::AtomicCell;
use ndarray::{Array, ArrayD, Ix1, IxDyn};
use std::alloc::{AllocInit, AllocRef, Global, Layout};
use std::cell::UnsafeCell;
use std::marker::PhantomPinned;
use std::pin::Pin;
use std::ptr::{self, NonNull, Unique};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use crate::game::Game;
use std::hash::{Hash, Hasher};

const MAX_SEGMENTS: usize = 32;

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
            win_scale: win_scale,
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

struct RawNodeStore<G: Game> {
    ptr: [UnsafeCell<Option<Unique<NodeLink<G>>>>; MAX_SEGMENTS],
    protected: [UnsafeCell<Option<Unique<AtomicBool>>>; MAX_SEGMENTS],
    segments: [AtomicUsize; MAX_SEGMENTS],
    num_segments: AtomicUsize,
    cap: AtomicUsize,
    growing: AtomicBool,
}

impl<G: Game> RawNodeStore<G> {
    fn new(reserve_space: usize) -> Self {
        let cap = reserve_space;
        assert!(cap > 1024);

        unsafe {
            let ptr = Global.alloc(
                Layout::array::<NodeLink<G>>(reserve_space).unwrap(),
                AllocInit::Uninitialized,
            );
            let ptr = Unique::new_unchecked(ptr.unwrap().ptr.as_ptr() as *mut _);

            let protected_ptr = Global.alloc(
                Layout::array::<AtomicBool>(reserve_space).unwrap(),
                AllocInit::Uninitialized,
            );
            let protected_ptr =
                Unique::new_unchecked(protected_ptr.unwrap().ptr.as_ptr() as *mut _);

            let mut ptrs: [UnsafeCell<_>; MAX_SEGMENTS] = array_init(|_| UnsafeCell::new(None));
            let mut protected_ptrs: [UnsafeCell<_>; MAX_SEGMENTS] =
                array_init(|_| UnsafeCell::new(None));
            ptrs[0] = UnsafeCell::new(Some(ptr));
            protected_ptrs[0] = UnsafeCell::new(Some(protected_ptr));

            let segments: [AtomicUsize; MAX_SEGMENTS] =
                array_init(|_| AtomicUsize::new(1 << (MAX_SEGMENTS + 1)));
            segments[0].store(reserve_space, Ordering::SeqCst);

            RawNodeStore {
                ptr: ptrs,
                protected: protected_ptrs,
                segments: segments,
                num_segments: AtomicUsize::new(1),
                cap: AtomicUsize::new(cap),
                growing: AtomicBool::new(false),
            }
        }
    }

    fn grow(&self) -> bool {
        if self
            .growing
            .compare_exchange_weak(false, true, Ordering::SeqCst, Ordering::Relaxed)
            .is_err()
        {
            return false;
        }

        let elem_size = std::mem::size_of::<NodeLink<G>>();
        assert!(elem_size != 0, "capacity overflow");

        let cap = self.cap.load(Ordering::SeqCst);
        let new_cap = 2 * cap;

        let segment_idx = self.num_segments.fetch_add(1, Ordering::SeqCst);

        unsafe {
            let new_ptr = Global.alloc(
                Layout::array::<NodeLink<G>>(cap).unwrap(),
                AllocInit::Uninitialized,
            );
            let new_ptr = Unique::new_unchecked(new_ptr.unwrap().ptr.as_ptr() as *mut _);
            *self.ptr[segment_idx].get() = Some(new_ptr);

            let new_protected_ptr =
                Global.alloc(Layout::array::<AtomicBool>(cap).unwrap(), AllocInit::Zeroed);

            let new_protected_ptr =
                Unique::new_unchecked(new_protected_ptr.unwrap().ptr.as_ptr() as *mut _);
            *self.protected[segment_idx].get() = Some(new_protected_ptr);
        }

        self.segments[segment_idx].store(2 * cap, Ordering::SeqCst);
        self.cap.store(new_cap, Ordering::SeqCst);
        self.growing.store(false, Ordering::SeqCst);

        true
    }
}

impl<G: Game> Drop for RawNodeStore<G> {
    // TODO: Fix potential race-condition memory leak on drop
    fn drop(&mut self) {
        let elem_size = std::mem::size_of::<NodeLink<G>>();
        self.growing.store(true, Ordering::SeqCst);
        let cap = self.cap.load(Ordering::SeqCst);
        if cap != 0 && elem_size != 0 {
            for i in 0..self.num_segments.load(Ordering::SeqCst) {
                let ptr = unsafe { self.ptr[i].get().as_ref() }.unwrap().unwrap();
                unsafe {
                    let c: NonNull<NodeLink<G>> = ptr.into();
                    Global.dealloc(c.cast(), Layout::array::<Node<G>>(cap).unwrap());
                }
            }
        }
    }
}

pub struct NodeStore<G: Game> {
    buf: RawNodeStore<G>,
    pub len: AtomicUsize,
    pub seen: DHashMap<G, usize>,
}

impl<G: Game> Drop for NodeStore<G> {
    fn drop(&mut self) {
        // spinlock till we drop
        loop {
            match self.buf.growing.compare_exchange_weak(
                false,
                true,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    break;
                }
                Err(_) => {}
            }
        }

        for i in 0..self.len.load(Ordering::SeqCst) {
            // drop everything first
            unsafe { drop(std::ptr::read(self.offset_ptr(i))) }
            unsafe { drop(std::ptr::read(self.offset_protected_ptr(i))) }
        }
    }
}

#[derive(PartialEq)]
pub enum NodeState {
    DoesNotExist,
    PlaceHolder,
    Locked,
    Exists(bool),
}

impl<G: Game> NodeStore<G> {
    fn get_segment_idx(&self, i: usize) -> usize {
        let num_segments = self.buf.num_segments.load(Ordering::SeqCst);
        debug_assert!(i < self.buf.segments[num_segments - 1].load(Ordering::SeqCst));
        let segment = match self
            .buf
            .segments
            .binary_search_by_key(&i, |segment_id| segment_id.load(Ordering::SeqCst))
        {
            Ok(segment) => segment + 1,
            Err(segment) => segment,
        };
        debug_assert!(segment < num_segments);
        segment
    }

    fn offset_ptr(&self, i: usize) -> *const NodeLink<G> {
        let segment_idx = self.get_segment_idx(i);

        let start_idx = if segment_idx > 0 {
            self.buf.segments[segment_idx - 1].load(Ordering::SeqCst)
        } else {
            0
        };

        unsafe {
            self.buf.ptr[segment_idx]
                .get()
                .as_ref()
                .unwrap()
                .unwrap()
                .as_ptr()
                .offset((i - start_idx) as isize)
        }
    }

    fn offset_ptr_mut(&self, i: usize) -> *mut NodeLink<G> {
        let segment_idx = self.get_segment_idx(i);

        let start_idx = if segment_idx > 0 {
            self.buf.segments[segment_idx - 1].load(Ordering::SeqCst)
        } else {
            0
        };

        unsafe {
            self.buf.ptr[segment_idx]
                .get()
                .as_ref()
                .unwrap()
                .unwrap()
                .as_ptr()
                .offset((i - start_idx) as isize)
        }
    }

    fn offset_protected_ptr(&self, i: usize) -> *const AtomicBool {
        let segment_idx = self.get_segment_idx(i);

        let start_idx = if segment_idx > 0 {
            self.buf.segments[segment_idx - 1].load(Ordering::SeqCst)
        } else {
            0
        };

        unsafe {
            self.buf.protected[segment_idx]
                .get()
                .as_ref()
                .unwrap()
                .unwrap()
                .as_ptr()
                .offset((i - start_idx) as isize)
        }
    }

    pub fn empty(reserve_space: usize) -> Self {
        NodeStore {
            buf: RawNodeStore::new(reserve_space),
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
            debug_assert!(idx < len);

            match unsafe { self.offset_ptr(l).as_ref() } {
                Some((_, None)) => return Some(l),
                Some((_, Some(s))) => l = *s,
                None => return None,
            };
        }
    }

    pub fn get(&self, idx: usize) -> Option<Pin<&Node<G>>> {
        let l = self.resolve(idx);

        l.and_then(|l| unsafe { Some(Pin::new_unchecked(&self.offset_ptr(l).as_ref().unwrap().0)) })
    }

    pub fn lookup_state_id(&self, s: &G) -> Option<usize> {
        self.seen.get(s).and_then(|s| Some(*s))
    }

    pub fn lookup_state(&self, s: &G) -> Option<Pin<&Node<G>>> {
        self.seen.get(s).and_then(|idx| self.get(*idx))
    }

    pub fn set_policy(&self, idx: usize, policy: Array<f32, Ix1>) -> bool {
        let l = self.resolve(idx);

        if self.state(idx) != NodeState::Locked {
            false
        } else {
            let r = unsafe { &mut self.offset_ptr_mut(l.unwrap()).as_mut().unwrap().0 };
            r.mu.p = Some(policy);

            true
        }
    }

    pub fn push(&self, node: Node<G>) -> usize {
        // grow the buffer once we've exceeded the halfway
        // point
        let idx = self.len.fetch_add(1, Ordering::SeqCst);
        if idx == self.buf.cap.load(Ordering::SeqCst) / 2 {
            self.buf.grow();
        }

        // in most cases this will never happen, but
        // we cannot continue until the buffer is big enough
        while idx >= self.buf.cap.load(Ordering::SeqCst) {}

        unsafe { ptr::write(self.offset_ptr(idx) as *mut NodeLink<G>, (node, None)) };

        // probably not needed because of our allocation strategy of zeroing?
        unsafe {
            self.offset_protected_ptr(idx)
                .as_ref()
                .unwrap()
                .store(false, Ordering::SeqCst)
        };

        idx
    }

    pub fn state(&self, idx: usize) -> NodeState {
        if idx < self.len() {
            let protected = unsafe { self.offset_protected_ptr(idx).as_ref() }.unwrap();
            if protected.load(Ordering::SeqCst) {
                NodeState::Locked
            } else {
                let nodelink = unsafe { self.offset_ptr(idx).as_ref() }.unwrap();
                if nodelink.1.is_none() {
                    if nodelink.0.mu.s.is_some() {
                        NodeState::Exists(true)
                    } else {
                        NodeState::PlaceHolder
                    }
                } else {
                    NodeState::Exists(false)
                }
            }
        } else {
            NodeState::DoesNotExist
        }
    }

    pub fn upgrade(&self, idx: usize, s: G) -> Option<bool> {
        let ptr = self.offset_ptr(idx) as *mut NodeLink<G>;

        if ptr.is_null() {
            None
        } else {
            let old_node = unsafe { &mut *ptr };
            let node_link: &mut Option<usize> = &mut old_node.1;

            if node_link.is_none() {
                let old_node = &mut old_node.0;
                let existing_idx = self.seen.get(&s);
                match existing_idx {
                    Some(existing_idx) => {
                        *node_link = Some(*existing_idx);
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
            } else {
                panic!("MCTS tried to upgrade already constructed Node!");
            }
        }
    }

    pub fn lock(&self, idx: usize) -> bool {
        let protected = unsafe { self.offset_protected_ptr(idx).as_ref() }.unwrap();
        match protected.compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed) {
            Err(_) => false,
            Ok(_) => true,
        }
    }

    pub fn unlock(&self, idx: usize) {
        unsafe {
            let protected = self.offset_protected_ptr(idx).as_ref().unwrap();
            let unlocked = protected
                .compare_exchange(true, false, Ordering::SeqCst, Ordering::Relaxed)
                .unwrap();
            debug_assert!(unlocked);
        }
    }

    fn sort_children(&self, buf: &mut Vec<(usize, f32)>, node: &Node<G>, cpuct: i32) {
        buf.clear();

        let parent_n = node.get_n();
        buf.extend(node.children.iter().map(|&child_idx| {
            let child = self.get(child_idx).unwrap();
            let u = child.compute_q()
                + (cpuct as f32)
                    * node.mu.p.as_ref().unwrap()[child.a as usize]
                    * f32::sqrt(parent_n as f32 + EPS)
                    / (1 + child.get_n()) as f32;
            (child_idx, u)
        }));

        buf.sort_by(|(_, u1), (_, u2)| u2.partial_cmp(u1).unwrap_or(std::cmp::Ordering::Equal));
    }

    pub fn best_child(&self, idx: usize, cpuct: i32, lock_leaf: bool) -> (usize, NodeState) {
        let node = self.get(idx).unwrap();

        let mut buf: Vec<(usize, f32)> = Vec::with_capacity(node.children.len());

        // keep looping until we get something!
        loop {
            // we could do this loop by checking if all the nodes
            // are valid first and then picking the best one. that
            // approach avoids the sorting algorithm, but it means that
            // we have to call NodeStore::state on _every_ child, which
            // is really not ideal given that NodeStore::state uses
            // atomics with sequential consistency.

            // another approach is to create a min binary heap and then
            // keep removing the top node as we go through the array so that
            // we only keep around the best k nodes at a time. if the binary
            // heap is stack-allocated, then there's no heap allocation.
            // the downside is that if all k elements are locked then there's
            // not much you can do other than keep looping. also we might
            // have games that have really large branching factors, in which
            // case stack allocation doesn't really help us

            self.sort_children(&mut buf, &node, cpuct);

            for (child_idx, _) in &buf {
                let child_state = self.state(*child_idx);
                // we know that all children must exist at least
                if child_state != NodeState::Locked {
                    if child_state == NodeState::PlaceHolder {
                        // try locking this node, but if it doesn't work,
                        // then just keep going
                        if !self.lock(*child_idx) {
                            continue;
                        }
                    }
                    return (*child_idx, child_state);
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }
}

unsafe impl<G: Game> Sync for NodeStore<G> {}

#[cfg(test)]
mod tests {
    mod dummy_game;
    use super::*;
    use dummy_game::DummyGame;
    use rayon::prelude::*;
    use std::collections::HashMap;

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
    fn test_nodestore_many() {
        let game = DummyGame::new(0);
        let nodes = NodeStore::<DummyGame>::empty(2048);

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
        let game = DummyGame::new(0);
        let nodes = NodeStore::<DummyGame>::empty(2048);

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
        let game = DummyGame::new(0);
        let nodes = NodeStore::<DummyGame>::empty(2048);

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
        let game = DummyGame::new(0);
        let nodes = NodeStore::<DummyGame>::empty(2048);

        let idx: HashMap<usize, usize> = (0..8192)
            .into_par_iter()
            .map(|i| {
                let node = Node::empty(10000.0f32);
                node.e.store(i as f32);

                // not really a great test case
                if i > 32 && nodes.state((i - 32) as usize) == NodeState::PlaceHolder {
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
        let nodes = NodeStore::<DummyGame>::empty(2048);

        let s = DummyGame::new(0);

        let idx: HashMap<usize, Option<usize>> = (0..8192)
            .into_iter()
            .map(|i| {
                let node = Node::empty(10000.0f32);
                let idx = nodes.push(node);
                assert!(nodes.lock(idx));
                let unique = nodes.upgrade(idx, s.clone()).unwrap();
                if unique {
                    assert!(nodes.state(idx) == NodeState::Locked);
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
        assert!(nodes.state(0) == NodeState::Locked);

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

        assert!(nodes.state(0) == NodeState::Exists(true));

        assert!(nodes.get(idx).unwrap().mu.s.clone().unwrap() == s);
        assert!(nodes.seen.contains_key(&s));
        assert!(nodes.seen.len() == 1);

        // insertion of s at location 1 creates a link, already seen
        nodes.push(node.clone());
        assert!(nodes.lock(1));
        assert!(!nodes.upgrade(1, s.clone()).unwrap());
        assert!(nodes.state(1) == NodeState::Exists(false));
    }

    #[test]
    fn test_nodestore_lock() {
        let nodes = NodeStore::<DummyGame>::empty(2048);

        let node = Node::empty(10000.0f32);
        let idx = nodes.push(node.clone());
        assert!(nodes.state(idx) == NodeState::PlaceHolder);

        let s = DummyGame::new(0);

        assert!(nodes.lock(idx));
        nodes.upgrade(idx, s.clone()).unwrap();

        // trying to lock a second time fails
        assert!(!nodes.lock(idx));
        assert!(nodes.state(idx) == NodeState::Locked);

        nodes.unlock(idx);

        // now it is unlocked
        assert!(nodes.state(idx) == NodeState::Exists(true));
    }
}
