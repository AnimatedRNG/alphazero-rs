use counter::Counter;
use log::{error, info};
use permutohedron::Heap;

use crate::game::Game;

pub fn play_game<G: Game>(
    player_actions: &[&dyn Fn(&G) -> u8],
    board: &Option<G>,
    verbose: bool,
) -> i8 {
    assert!(player_actions.len() == 2);

    let mut cur_player = 1;
    let mut board: G = board.clone().unwrap_or_else(G::get_init_board);
    let mut iteration = 0;

    while board.get_game_ended(cur_player) == 0.0 {
        iteration += 1;

        if verbose {
            info!("Turn {}, Player {}\n{}", iteration, cur_player, board);
        }

        let canonical_board = board.get_canonical_form(cur_player);

        let action = player_actions[if cur_player == 1 { 0 } else { 1 }](&canonical_board);

        let valids = canonical_board.get_valid_moves(1);

        if valids[action as usize] == 0 {
            error!("Action {} is not valid!", action);
            error!("valids = {}", valids);
            assert!(valids[action as usize] > 0);
        }

        let (new_board, new_cur_player) = board.get_next_state(cur_player, action);
        board = new_board;
        cur_player = new_cur_player;
    }

    if verbose {
        info!(
            "Game over on turn {}; Result: {}",
            iteration,
            board.get_game_ended(1)
        )
    }

    // if it's close to 0, just round it to a draw
    cur_player * f32::round(board.get_game_ended(cur_player)) as i8
}

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
pub enum GameResult {
    Win,
    Loss,
    Draw,
}

// TODO: Fix for n-player games
pub fn play_games<G: Game>(
    num: usize,
    player_actions: Vec<&dyn Fn(&G) -> u8>,
    board: Option<G>,
    verbose: bool,
) -> Counter<GameResult, usize> {
    assert!(player_actions.len() == 2);

    let mut player_actions: Vec<_> = player_actions.into_iter().enumerate().collect();

    let mut all_counts: Counter<GameResult, usize> = Counter::new();

    Heap::new(&mut player_actions)
        .into_iter()
        .for_each(|player_ordering| {
            let player_actions: Vec<_> = player_ordering.iter().map(|(_, a)| *a).collect();
            //let win_idx = player_ordering.iter().position(|(i, _)| *i == 0).unwrap();
            //let loss_idx = player_ordering.iter().position(|(i, _)| *i == 1).unwrap();
            let win_cond = if player_ordering[0].0 == 0 { 1 } else { -1 };
            let lose_cond = if player_ordering[0].0 == 0 { -1 } else { 1 };

            let counts: Counter<GameResult, usize> = (0..(num / 2))
                .map(|_| {
                    let game_result = play_game(&player_actions, &board, verbose);
                    if game_result == win_cond {
                        GameResult::Win
                    } else if game_result == lose_cond {
                        GameResult::Loss
                    } else {
                        GameResult::Draw
                    }
                })
                .collect();
            all_counts.extend(&counts);
        });

    all_counts
}
