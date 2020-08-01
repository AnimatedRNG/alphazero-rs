use ndarray::{ArcArray, Array, ArrayD, ArrayView, ArrayViewD, Ix1, Ix2, IxDyn};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::game::F;

pub type ArcArrayD<I> = ArcArray<I, IxDyn>;

pub type BoardFeatures = ArrayD<F>;
pub type BoardFeaturesView<'a> = ArrayViewD<'a, F>;
pub type BatchedBoardFeatures = ArrayD<F>;
pub type BatchedBoardFeaturesView<'a> = ArrayViewD<'a, F>;
pub type SerializedBoardFeatures = Vec<F>;
pub type BatchedPolicy = Array<f32, Ix2>;
pub type BatchedPolicyView<'a> = ArrayView<'a, f32, Ix2>;
pub type Policy = Array<f32, Ix1>;
pub type PolicyView<'a> = ArrayView<'a, f32, Ix1>;
pub type SerializedPolicy = Vec<f32>;
pub type BatchedValue = Array<f32, Ix1>;
pub type Value = f32;

#[derive(Serialize, Deserialize)]
pub struct TrainingSample {
    pub board: BoardFeatures,
    pub pi: Policy,
    pub v: Value,
}

pub type ArcBatchedBoardFeatures = ArcArrayD<F>;
pub type ArcBatchedPolicy = ArcArray<f32, Ix2>;
pub type ArcBatchedValue = ArcArray<f32, Ix1>;

pub type SOATrainingSamples<'a> = (
    ArcBatchedBoardFeatures,
    ArcBatchedPolicy,
    ArcBatchedValue,
);

pub trait NNet: Send + Clone + 'static {
    fn new() -> Self;

    //fn train(&self, examples: Vec<TrainingSample>);
    fn train(&self, examples: SOATrainingSamples);

    fn predict(&self, board: BatchedBoardFeaturesView) -> (BatchedPolicy, BatchedValue);

    fn save_checkpoint<P: AsRef<Path>>(&self, path: P);

    fn load_checkpoint<P: AsRef<Path>>(&mut self, path: &P);
}
