use ndarray::{Array, ArrayD, Ix1};
use std::cmp::Eq;
use std::fmt::Display;
use std::hash::Hash;

use crate::nnet::{Policy, PolicyView};

pub type F = f32;

pub trait Game: Display + Sized + Send + Clone + Hash + Eq {
    fn get_init_board() -> Self;

    fn get_feature_shape() -> Vec<usize>;

    fn get_next_state(&self, player: i8, action: u8) -> (Self, i8);

    fn get_valid_moves(&self, player: i8) -> Array<u8, Ix1>;

    fn get_game_ended(&self, player: i8) -> f32;

    fn get_canonical_form(&self, player: i8) -> Self;

    fn get_symmetries(&self, pi: PolicyView) -> Vec<(Self, Policy)>;

    fn eval_heuristic(&self) -> f32;

    fn to_features(&self) -> ArrayD<F>;
}
