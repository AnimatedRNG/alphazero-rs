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

pub struct ConnectFourGame {
    s: [[i8; DEFAULT_HEIGHT]; DEFAULT_WIDTH],
    heights: [usize; DEFAULT_WIDTH],
    me: i8,
}

impl Clone for ConnectFourGame {
    fn clone(&self) -> ConnectFourGame {
        ConnectFourGame {
            s: self.s,
            heights: self.heights,
            me: self.me,
        }
    }
}

impl fmt::Display for ConnectFourGame {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut rows: Vec<_> = Vec::new();
        for i in 0..DEFAULT_HEIGHT {
            let mut cols: Vec<_> = Vec::new();
            for j in 0..DEFAULT_WIDTH {
                cols.push(match self.s[i][j] {
                    0 => "_",
                    1 => "1",
                    -1 => "2",
                    _ => panic!("invalid value in board"),
                });
            }
            rows.push(cols.join(" "));
        }

        let text: String = rows.join("\n");
        write!(f, "{}", text)
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
        debug_assert!(height < &mut DEFAULT_HEIGHT);
        *height += 1;
        next_state.s[action][*height] = player;

        (ConnectFourGame::empty(), 1 - player)
    }

    fn get_valid_moves(&self, _: i8) -> Array<u8, Ix1> {
        (0..DEFAULT_WIDTH as u8)
            .map(|j| (j, self.heights[usize::from(j)]))
            .filter(|(_, k)| k < &DEFAULT_HEIGHT)
            .map(|(j, _)| j)
            .collect()
    }

    fn get_game_ended(&self, player: i8) -> i8 {
        let mut running_sum: usize = 0;
        let check = |i: usize, j: usize, running_sum: &mut usize| {
            *running_sum = if self.s[i][j] == player {
                *running_sum + 1
            } else {
                0
            };

            if running_sum >= &mut DEFAULT_WIN_LENGTH {
                Some(player)
            } else {
                None
            }
        };

        // left/right
        for i in 0..DEFAULT_HEIGHT {
            running_sum = 0;
            for j in 0..DEFAULT_WIDTH {
                if check(i, j, &mut running_sum).is_some() {
                    return player;
                }
            }
        }

        // up/down
        for j in 0..DEFAULT_WIDTH {
            for i in 0..DEFAULT_HEIGHT {
                if check(i, j, &mut running_sum).is_some() {
                    return player;
                }
            }
        }

        // diagonal +1/+1
        for k in 0..DEFAULT_WIDTH + DEFAULT_HEIGHT - 2 {
            for j in 0..k {
                let i = k - j;
                if i < DEFAULT_HEIGHT
                    && j < DEFAULT_WIDTH
                    && check(i, j, &mut running_sum).is_some()
                {
                    return player;
                }
            }
        }

        // diagonal +1/-1
        for k in 0..DEFAULT_WIDTH + DEFAULT_HEIGHT - 2 {
            for j in (0..k).rev() {
                let i = k - j;
                if i < DEFAULT_HEIGHT
                    && j < DEFAULT_WIDTH
                    && check(i, j, &mut running_sum).is_some()
                {
                    return player;
                }
            }
        }

        0
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
        let mut f = ArrayD::zeros(IxDyn(&[2, DEFAULT_HEIGHT, DEFAULT_WIDTH]));

        f.indexed_iter_mut().for_each(|(tup, v)| {
            let c = tup[0];
            let i = tup[1];
            let j = tup[2];

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
