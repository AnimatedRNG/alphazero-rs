use ndarray::{Array, ArrayD, ArrayView, ArrayViewD, Ix1};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::game::F;

pub type BoardFeatures = ArrayD<F>;
pub type BoardFeaturesView<'a> = ArrayViewD<'a, F>;
pub type SerializedBoardFeatures = Vec<F>;
pub type Policy = Array<f32, Ix1>;
pub type PolicyView<'a> = ArrayView<'a, f32, Ix1>;
pub type SerializedPolicy = Vec<f32>;
pub type Value = f32;

#[derive(Serialize, Deserialize)]
pub struct TrainingSample {
    pub board: BoardFeatures,
    pub pi: Policy,
    pub v: Value,
}

pub trait NNet: Send + Clone + 'static {
    fn new() -> Self;

    fn train(&self, examples: Vec<TrainingSample>);

    fn predict(&self, board: BoardFeaturesView) -> (Policy, Value);

    fn save_checkpoint<P: AsRef<Path>>(&self, path: P);

    fn load_checkpoint<P: AsRef<Path>>(&mut self, path: &P);
}
