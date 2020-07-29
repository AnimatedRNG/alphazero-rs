use crate::game::{Game, F};
use std::hash::{Hash, Hasher};
use std::fmt;

use ndarray::{Array, ArrayViewD, ArrayD, Ix1, IxDyn};

use crate::nnet::{Policy, PolicyView};
//pub type Policy = Array<f32, Ix1>;
//pub type PolicyView = Array<f32, Ix1>;

pub struct DummyGame {
    _s: u8
}

impl Clone for DummyGame {
    fn clone(&self) -> DummyGame {
        DummyGame::new(self._s)
    }
}

impl fmt::Display for DummyGame {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "")
    }
}

impl Hash for DummyGame {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self._s.hash(hasher);
    }
}

impl PartialEq for DummyGame {
    fn eq(&self, other: &Self) -> bool {
        self._s == other._s
    }
}

impl Eq for DummyGame {}

impl DummyGame {
    pub fn new(v: u8) -> DummyGame {
        DummyGame {_s: v}
    }
}

impl Game for DummyGame {
    fn get_init_board() -> DummyGame {
        DummyGame::new(0)
    }

    fn get_feature_shape() -> Vec<usize> {
        vec![1]
    }

    fn get_next_state(
        &self,
        player: i8,
        action: u8,
    ) -> (DummyGame, i8) {
        (DummyGame::new(self._s + 1), 1 - player)
    }

    fn get_valid_moves(&self, player: i8) -> Array<u8, Ix1> {
        Array::zeros(1)
    }

    fn get_game_ended(&self, player: i8) -> i8 {
        0
    }

    fn get_canonical_form(&self, player: i8) -> DummyGame {
        DummyGame::new(0)
    }

    fn get_symmetries(
        &self,
        pi: PolicyView,
    ) -> Vec<(Self, Policy)> {
        vec![(DummyGame::new(self._s), pi.to_owned())]
    }

    fn eval_heuristic(&self) -> f32 {
        0.0
    }

    fn to_features(&self) -> ArrayD<F> {
        ArrayD::from_elem(IxDyn(&[1]), self._s as F)
    }
}
