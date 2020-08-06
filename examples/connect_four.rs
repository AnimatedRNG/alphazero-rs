use alphazero_rs::nnet::NNet;
use ndarray::Array;

mod connect_four_lib;
mod utils;

use connect_four_lib::connect_four_game::ConnectFourGame;

struct DumbConnectFourNnet;

impl NNet for DumbConnectFourNnet {
    fn new<P: AsRef<std::path::Path>>(_checkpoint: P) -> Self {
        Self
    }
    fn train(
        &mut self,
        _examples: alphazero_rs::nnet::SOATrainingSamples,
        _previous_model_id: usize,
        _model_id: usize,
    ) {
    }

    fn predict(
        &self,
        board: alphazero_rs::nnet::BatchedBoardFeaturesView,
        _model_id: usize,
    ) -> (
        alphazero_rs::nnet::BatchedPolicy,
        alphazero_rs::nnet::BatchedValue,
    ) {
        if let [batches, width, _height] = *board.shape() {
            (Array::ones((batches, width)), Array::ones((batches,)))
        } else {
            panic!()
        }
    }
}

fn ask_for_action(game: &ConnectFourGame) -> u8 {
    println!("{}", game);
    let reader = std::io::stdin();
    let input = &mut String::new();
    reader.read_line(input).unwrap();
    input.trim().parse().unwrap()
}

fn main() {
    pretty_env_logger::init();
    alphazero_rs::arena::play_game(&[&ask_for_action, &ask_for_action], &None, true);
}
