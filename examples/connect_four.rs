use alphazero_rs::coach::Coach;
use alphazero_rs::nnet::NNet;
use ndarray::Array;
use rand::rngs::SmallRng;
use rand::SeedableRng;

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
        if let [batches, _height, width] = *board.shape() {
            (
                Array::ones((batches, width)) / width as f32,
                Array::ones((batches,)),
            )
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
    let mut coach = Coach::setup(
        "./checkpoint", // checkpoint_directory
        1000000,        // mcts_reserve_size
        0.6,            // update_threshold
        15,             // temp_threshold
        20,             // max_history_length
        200000,         // max_queue_length
        1,              // inference_batch_size
        1,              // num_episode_threads
        40,             // num_arena_games
        1,              // num_iters
        1,              // num_eps
        25,             // num_sims
        1,              // num_sim_threads,
        1000,           // max_depth
        1,              // cpuct
    );
    coach.learn::<ConnectFourGame, DumbConnectFourNnet, String>(
        "./checkpoint".to_string(),
        false,
        true,
        &mut SmallRng::from_entropy(),
    );

    //alphazero_rs::arena::play_game(&[&ask_for_action, &ask_for_action], &None, true);
}
