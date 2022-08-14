use pyo3::prelude::*;
// use pyo3::types::*;
use numpy::*;
use ndarray::*;
// use std::slice;
// use std::ops::Index;

/// Formats the sum of two numbers as string.
// #[pyfunction]
// fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
//     Ok((a + b).to_string())
// }

/// Some RLGym functions that were converted into Rust
#[pymodule]
fn rlgym_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(quat_to_rot_mtx, m)?)?;
    m.add_function(wrap_pyfunction!(norm_func, m)?)?;
    // m.add_class::<GameState>()?;
    // m.add_class::<PhysicsObject>()?;
    // m.add_class::<PlayerData>()?;
    Ok(())
}

/// Quat to rot matrix calculation for RLGym using list
#[pyfunction]
#[pyo3(text_signature = "Takes quat list and outputs matrix in ndarray type")]
fn quat_to_rot_mtx(nums: Vec<f64>, py: Python) -> PyResult<&PyArray2<f64>> {
    let mut theta = Array2::<f64>::zeros((3, 3));

    let norm: f64 = nums.clone()
                        .into_iter()
                        .map(|x: f64| x.powf(2.))
                        .collect::<Vec<f64>>()
                        .iter()
                        .sum();
    let s: f64 = 1.0 / norm;

    let w: &f64 = &nums[0];
    let x: &f64 = &nums[1];
    let y: &f64 = &nums[2];
    let z: &f64 = &nums[3];

    // front direction
    theta[[0, 0]] = 1. - 2. * s * (y * y + z * z);
    theta[[1, 0]] = 2. * s * (x * y + z * w);
    theta[[2, 0]] = 2. * s * (x * z - y * w);

    // left direction
    theta[[0, 1]] = 2. * s * (x * y - z * w);
    theta[[1, 1]] = 1. - 2. * s * (x * x + z * z);
    theta[[2, 1]] = 2. * s * (y * z + x * w);

    // up direction
    theta[[0, 2]] = 2. * s * (x * z + y * w);
    theta[[1, 2]] = 2. * s * (y * z - x * w);
    theta[[2, 2]] = 1. - 2. * s * (x * x + y * y);

    // let theta_arr = theta.to_pyarray(py);

    Ok(theta.to_pyarray(py))
}

/// Norm func that takes list
#[pyfunction]
#[pyo3(text_signature = "takes list and norms it")]
fn norm_func(nums: Vec<f64>) -> PyResult<f64> {
    let norm_val: f64 = nums.clone()
                                 .into_iter()
                                 .map(|x: f64| x.powf(2.))
                                 .collect::<Vec<f64>>()
                                 .iter()
                                 .sum::<f64>()
                                 .sqrt();
    Ok(norm_val)
}


// eh this probably isn't worth it honestly
// #[pyfunction]
// #[pyo3(text_signature = "takes list of actions and clips it")]
// fn clip_func(actions: Vec<f64>, high: f64, low: f64) -> PyResult<f64> {
//     let clipped_actions: f64 = actions.clone()
//                                       .into_iter()
//                                       .map(|x: f64| if x > high {high} else if x < low {low} else {x})
//                                       .collect::<Vec<f64>>();
//     Ok(clipped_actions)
// }

// const GS_BOOST_PADS_LENGTH: i32 = 34;
// const GS_BALL_STATE_LENGTH: i32 = 18;
// const GS_PLAYER_CAR_STATE_LENGTH: i32 = 13;
// const GS_PLAYER_TERTIARY_INFO_LENGTH: i32 = 11;
// const GS_PLAYER_INFO_LENGTH: i32 = 2 + 2 * GS_PLAYER_CAR_STATE_LENGTH * GS_PLAYER_TERTIARY_INFO_LENGTH;
// GameState struct
// #[pyclass]
// struct GameState {
//     game_type: i64,
//     blue_score: i64,
//     orange_score: i64,
//     last_touch: i64,
//     players: Vec<PlayerData>,
//     ball: PhysicsObject,
//     inverted_ball: PhysicsObject,
//     boost_pads: Vec<i64>,
//     inverted_boost_pads: Vec<i64>
// }

// PhysicsObject struct
// #[pyclass]
// struct PhysicsObject {
//     position: Array1<f64>,
//     quaternion: Array1<f64>,
//     linear_velocity: Array1<f64>,
//     angular_velocity: Array1<f64>,
//     euler_angles: Array1<f64>,
//     rotation_mtx: Array2<f64>,
//     has_computed_rot_mtx: bool,
//     has_computed_euler_angles: bool
// }

// PlayerData struct
// #[pyclass]
// struct PlayerData {
//     car_id: i32,
//     team_num: i32,
//     match_goals: i64,
//     match_saves: i64,
//     match_shots: i64,
//     match_demolishes: i64,
//     boost_pickups: i64,
//     is_demoed: bool,
//     on_ground: bool,
//     ball_touched: bool,
//     has_jump: bool,
//     has_flip: bool,
//     boost_amount: f64,
//     car_data: PhysicsObject,
//     inverted_car_data: PhysicsObject
// }

// fn decode_player(full_player_data: Vec<f64>) -> PlayerData {
//     let mut player_data: PlayerData = PlayerData::create_default();

//     let mut start: i32 = 2;

//     let car_data: Vec<f64> = full_player_data[start as usize..(start + GS_PLAYER_CAR_STATE_LENGTH) as usize].to_vec();
//     player_data.car_data.decode_car_data(car_data);
//     start = start + GS_PLAYER_CAR_STATE_LENGTH;

//     let inv_car_data: Vec<f64> = full_player_data[start as usize..(start + GS_PLAYER_CAR_STATE_LENGTH) as usize].to_vec();
//     player_data.inverted_car_data.decode_car_data(inv_car_data);
//     start = start + GS_PLAYER_CAR_STATE_LENGTH;

//     let tertiary_data: Vec<f64> = full_player_data[start as usize..(start + GS_PLAYER_TERTIARY_INFO_LENGTH) as usize].to_vec();

//     player_data.match_goals = tertiary_data[0] as i64;
//     player_data.match_saves = tertiary_data[1] as i64;
//     player_data.match_shots = tertiary_data[2] as i64;
//     player_data.match_demolishes = tertiary_data[3] as i64;
//     player_data.boost_pickups = tertiary_data[4] as i64;
//     player_data.is_demoed = tertiary_data[5] > 0.0;
//     player_data.on_ground = tertiary_data[6] > 0.0;
//     player_data.ball_touched = tertiary_data[7] > 0.0;
//     player_data.has_jump = tertiary_data[8] > 0.0;
//     player_data.has_flip = tertiary_data[9] > 0.0;
//     player_data.boost_amount = tertiary_data[10];
//     player_data.car_id = full_player_data[0] as i32;
//     player_data.team_num = full_player_data[1] as i32;

//     player_data
// }

// #[pymethods]
// impl GameState {
//     /// default/basic structure of GameState
//     #[new]
//     fn create_default() -> PyResult<GameState> {
//         Ok(Self { 
//             game_type: 0, 
//             blue_score: -1, 
//             orange_score: -1, 
//             last_touch: -1, 
//             players: Vec::new(), 
//             ball: PhysicsObject::create_default(), 
//             inverted_ball: PhysicsObject::create_default(), 
//             boost_pads: Vec::new(), 
//             inverted_boost_pads: Vec::new() 
//         })

//         //Ok(game_state)
//     }

// #[pyfunction]
// fn decode(state_floats: Vec<f64>) -> PyResult<Vec<>> {
//     let mut start: i32 = 3;
//     let num_ball_packets: i32 = 1;
//     let num_player_packets: i32 = (state_floats.len() as i32) - num_ball_packets * GS_BALL_STATE_LENGTH
//         - start - GS_BALL_STATE_LENGTH / GS_PLAYER_INFO_LENGTH;

//     self.blue_score = state_floats[1] as i64;
//     self.orange_score = state_floats[2] as i64;
//     self.boost_pads = state_floats[start as usize..(start + GS_BOOST_PADS_LENGTH) as usize].to_vec()
//                                                                                             .into_iter()
//                                                                                             .map(|x: f64| x as i64)
//                                                                                             .collect::<Vec<i64>>();
//     self.inverted_boost_pads = state_floats[(start + GS_BOOST_PADS_LENGTH) as usize..start as usize].to_vec()
//                                                                                                     .into_iter()
//                                                                                                     .map(|x: f64| x as i64)
//                                                                                                     .collect::<Vec<i64>>();
//     start = start + GS_BOOST_PADS_LENGTH;

//     let ball_data: Vec<f64> = state_floats[start as usize..(start + GS_BALL_STATE_LENGTH) as usize].to_vec();
//     self.ball.decode_ball_data(ball_data);
//     start = start + (GS_BALL_STATE_LENGTH / 2);

//     let inv_ball_data: Vec<f64> = state_floats[start as usize..(start + GS_BALL_STATE_LENGTH) as usize].to_vec();
//     self.inverted_ball.decode_ball_data(inv_ball_data);
//     start = start + (GS_BALL_STATE_LENGTH / 2);

//     // let mut player_vec: Vec<PlayerData> = Vec::new();
//     for _ in 0..num_player_packets as usize {
//         let player = decode_player(state_floats[start as usize..(start + GS_PLAYER_INFO_LENGTH) as usize].to_vec());
//         if player.ball_touched {
//             self.last_touch = player.car_id as i64;
//         }
//         self.players.push(player);

//         start = start + GS_PLAYER_INFO_LENGTH;
//     }
//     self.players.sort_by_key(|k| k.car_id);

//     Ok(())
// }

// #[pymethods]
// impl PlayerData {
//     /// default/basic structure of PlayerData
//     #[new]
//     fn create_default() -> Self {
//         let player_data: PlayerData = PlayerData {
//             car_id: -1,
//             team_num: -1,
//             match_goals: -1,
//             match_saves: -1,
//             match_shots: -1,
//             match_demolishes: -1,
//             boost_pickups: -1,
//             is_demoed: false,
//             on_ground: false,
//             ball_touched: false,
//             has_jump: false,
//             has_flip: false,
//             boost_amount: -1.,
//             car_data: PhysicsObject::create_default(),
//             inverted_car_data: PhysicsObject::create_default()
//         };

//         player_data
//     }
// }

// #[pymethods]
// impl PhysicsObject {
//     /// default/basic structure of PhysicsObject
//     #[new]
//     fn create_default() -> PyResult<PhysicsObject> {
//         Ok(Self {
//             position: Array1::<f64>::zeros(3),
//             quaternion: Array1::<f64>::ones(4),
//             linear_velocity: Array1::<f64>::ones(3),
//             angular_velocity: Array1::<f64>::zeros(3),
//             euler_angles: Array1::<f64>::zeros(3),
//             rotation_mtx: Array2::<f64>::zeros((3, 3)),
//             has_computed_rot_mtx: false,
//             has_computed_euler_angles: false
//             })
//         }
    
//     #[classmethod]
//     fn decode_car_data(&mut self, car_data: Vec<f64>) {
//         // fn exm(sli: &[f64]) -> [f64; 3] {
//         //     let arr = <[f64; 3]>::try_from(sli).unwrap();
//         //     arr
//         // }
//         let mut i = 0;
//         for x in car_data {
//             if i > 3 {
//                 self.position[i] = x
//             }
//             else if i < 6 && i >= 3 {
//                 self.linear_velocity[i] = x
//             }
//             else if i < 9 && i >= 6 {
//                 self.angular_velocity[i] = x
//             }
//             else {
//                 panic!("decode_car_data for loop went wrong, too many values from car_data to unpack")
//             }
//             i += 1;
//         }
//     }

//     #[classmethod]
//     fn decode_ball_data(&mut self, ball_data: Vec<f64>) {
//         // fn exm(sli: &[f64]) -> [f64; 3] {
//         //     let arr = <[f64; 3]>::try_from(sli).unwrap();
//         //     arr
//         // }
//         let mut i = 0;
//         for x in ball_data {
//             if i > 3 {
//                 self.position[i] = x
//             }
//             else if i < 7 && i >= 3 {
//                 self.quaternion[i] = x
//             }
//             else if i < 10 && i >= 7 {
//                 self.linear_velocity[i] = x
//             }
//             else if i < 14 && i >= 10 {
//                 self.angular_velocity[i] = x
//             }
//             else {
//                 panic!("decode_ball_data for loop went wrong, too many values from ball_data to unpack")
//             }
//             i += 1;
//         }
//     }
// }

// fn test() {
//     let mut play_dat: PlayerData = PlayerData::create_default();
//     let ang_vel: Array1<f64> = play_dat.car_data.angular_velocity;
//     play_dat.car_data.angular_velocity = Array1::<f64>::ones(3);
// }
