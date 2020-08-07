extern crate alphazero_rs;
extern crate array_init;

use alphazero_rs::game::{Game, F};
use alphazero_rs::nnet::{Policy, PolicyView};
use std::fmt;
use std::hash::{Hash, Hasher};

use array_init::array_init;

use ndarray::{s, Array, ArrayD, Ix1, IxDyn};

const DEFAULT_HEIGHT: usize = 6;
const DEFAULT_WIDTH: usize = 7;
const DEFAULT_WIN_LENGTH: usize = 4;
const DRAW_EPS: f32 = 1e-4;

#[derive(Clone, Debug)]
pub struct ConnectFourGame {
    s: [[i8; DEFAULT_WIDTH]; DEFAULT_HEIGHT],
    heights: [usize; DEFAULT_WIDTH],
    me: i8,
}

impl fmt::Display for ConnectFourGame {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for row in 0..DEFAULT_HEIGHT {
            for col in 0..DEFAULT_WIDTH {
                f.write_str(match self.s[row][col] {
                    0 => "_",
                    1 => "1",
                    -1 => "2",
                    _ => panic!("invalid value in board"),
                })?;
            }
            f.write_str("\n")?;
        }
        Ok(())
    }
}

impl Hash for ConnectFourGame {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.s.hash(hasher);
    }
}

impl PartialEq for ConnectFourGame {
    fn eq(&self, other: &Self) -> bool {
        self.s == other.s
    }
}

impl Eq for ConnectFourGame {}

impl ConnectFourGame {
    pub fn empty() -> ConnectFourGame {
        ConnectFourGame {
            s: array_init(|_| array_init(|_| 0)),
            heights: array_init(|_| 0),
            me: 1,
        }
    }

    pub fn flip(&self) -> ConnectFourGame {
        let mut cl = ConnectFourGame::empty();
        for i in 0..DEFAULT_HEIGHT {
            for j in 0..DEFAULT_WIDTH {
                cl.s[i][j] = self.s[i][DEFAULT_WIDTH - j - 1];
            }
        }

        for j in 0..DEFAULT_WIDTH {
            cl.heights[j] = self.heights[DEFAULT_WIDTH - j - 1];
        }

        cl
    }
}

impl Game for ConnectFourGame {
    fn get_init_board() -> ConnectFourGame {
        ConnectFourGame::empty()
    }

    fn get_feature_shape() -> Vec<usize> {
        vec![2, DEFAULT_HEIGHT, DEFAULT_WIDTH]
    }

    fn get_next_state(&self, player: i8, action: u8) -> (ConnectFourGame, i8) {
        let mut next_state = self.clone();

        let action = action as usize;

        let height = &mut next_state.heights[action];
        log::debug!("C4 move: {} {}", player, action);
        debug_assert!(height < &mut DEFAULT_HEIGHT);
        *height += 1;
        next_state.s[DEFAULT_HEIGHT - *height][action] = player;

        (next_state, -player)
    }

    fn get_valid_moves(&self, _: i8) -> Array<u8, Ix1> {
        (0..DEFAULT_WIDTH as u8)
            .map(|col| (col, self.heights[usize::from(col)]))
            .map(|(_, height)| (height < DEFAULT_HEIGHT) as u8)
            .collect()
    }

    fn get_game_ended(&self, player: i8) -> f32 {
        // left/right
        for row in 0..DEFAULT_HEIGHT {
            for col in 0..DEFAULT_WIDTH - DEFAULT_WIN_LENGTH {
                if self.s[row][col] != 0
                    && [self.s[row][col]; DEFAULT_WIN_LENGTH]
                        == self.s[row][col..col + DEFAULT_WIN_LENGTH]
                {
                    return if player == self.s[row][col] {
                        1f32
                    } else {
                        -1f32
                    };
                }
            }
        }

        // up/down
        for row in 0..DEFAULT_HEIGHT - DEFAULT_WIN_LENGTH {
            for col in 0..DEFAULT_WIDTH {
                if self.s[row][col] != 0
                    && [self.s[row][col]; DEFAULT_WIN_LENGTH]
                        == [
                            self.s[row][col],
                            self.s[row + 1][col],
                            self.s[row + 2][col],
                            self.s[row + 3][col],
                        ]
                {
                    return if player == self.s[row][col] {
                        1f32
                    } else {
                        -1f32
                    };
                }
            }
        }

        // diagonal +1/+1
        for row in 0..=DEFAULT_HEIGHT - DEFAULT_WIN_LENGTH {
            for col in 0..=DEFAULT_WIDTH - DEFAULT_WIN_LENGTH {
                if self.s[row][col] != 0
                    && [self.s[row][col]; DEFAULT_WIN_LENGTH]
                        == [
                            self.s[row][col],
                            self.s[row + 1][col + 1],
                            self.s[row + 2][col + 2],
                            self.s[row + 3][col + 3],
                        ]
                {
                    return if player == self.s[row][col] {
                        1f32
                    } else {
                        -1f32
                    };
                }
            }
        }

        // diagonal +1/-1
        for row in 0..=DEFAULT_HEIGHT - DEFAULT_WIN_LENGTH {
            for col in DEFAULT_WIN_LENGTH - 1..DEFAULT_WIDTH {
                if self.s[row][col] != 0
                    && [self.s[row][col]; DEFAULT_WIN_LENGTH]
                        == [
                            self.s[row][col],
                            self.s[row + 1][col - 1],
                            self.s[row + 2][col - 2],
                            self.s[row + 3][col - 3],
                        ]
                {
                    return if player == self.s[row][col] {
                        1f32
                    } else {
                        -1f32
                    };
                }
            }
        }

        if self.heights.iter().filter(|&h| h < &DEFAULT_HEIGHT).count() == 0 {
            DRAW_EPS
        } else {
            0.0
        }
    }

    fn get_canonical_form(&self, _player: i8) -> ConnectFourGame {
        let mut new_board = self.clone();
        new_board.me = -new_board.me;

        new_board
    }

    fn get_symmetries(&self, pi: PolicyView) -> Vec<(Self, Policy)> {
        // add flipped case
        vec![
            (self.clone(), pi.to_owned()),
            (self.clone().flip(), pi.slice(s![..;-1]).to_owned()),
        ]
    }

    // no heuristic
    fn eval_heuristic(&self) -> f32 {
        0.0
    }

    #[allow(clippy::if_same_then_else)]
    fn to_features(&self) -> ArrayD<F> {
        let mut f = ArrayD::zeros(IxDyn(&[DEFAULT_HEIGHT, DEFAULT_WIDTH, 2]));

        f.indexed_iter_mut().for_each(|(tup, v)| {
            let i = tup[0];
            let j = tup[1];
            let c = tup[2];

            *v = if self.s[i][j] == self.me && c == 0 {
                1.0 as F
            } else if self.s[i][j] == -self.me && c == 1 {
                1.0 as F
            } else {
                0.0 as F
            }
        });

        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_win_diagonal() {
        let board = ConnectFourGame::empty();
        let player = 1;
        let (board, player) = board.get_next_state(player, 0);

        let (board, player) = board.get_next_state(player, 1);
        let (board, player) = board.get_next_state(player, 1);

        let (board, player) = board.get_next_state(player, 2);
        let (board, player) = board.get_next_state(player, 0);
        let (board, player) = board.get_next_state(player, 2);
        let (board, player) = board.get_next_state(player, 2);

        let (board, player) = board.get_next_state(player, 3);
        let (board, player) = board.get_next_state(player, 3);
        let (board, player) = board.get_next_state(player, 3);
        let (board, player) = board.get_next_state(player, 3);

        assert_eq!(board.get_game_ended(1), 1f32);
    }
}
