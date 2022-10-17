# import math
from math import pi
# import numpy as np
from numpy import ndarray, zeros, expand_dims
from numba import njit
from typing import Any
from rlgym.utils import common_values
from rlgym.utils.gamestates import PlayerData, GameState, PhysicsObject
from rlgym.utils.obs_builders import ObsBuilder


# class AdvancedObs(ObsBuilder):
#     POS_STD = 2300  # If you read this and wonder why, ping Rangler in the dead of night.
#     ANG_STD = math.pi
#
#     def __init__(self):
#         super().__init__()
#
#     def reset(self, initial_state: GameState):
#         pass
#
#     def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
#
#         if player.team_num == common_values.ORANGE_TEAM:
#             inverted = True
#             ball = state.inverted_ball
#             pads = state.inverted_boost_pads
#         else:
#             inverted = False
#             ball = state.ball
#             pads = state.boost_pads
#
#         obs = [ball.position / self.POS_STD,
#                ball.linear_velocity / self.POS_STD,
#                ball.angular_velocity / self.ANG_STD,
#                previous_action,
#                pads]
#
#         player_car = self._add_player_to_obs(obs, player, ball, inverted)
#
#         allies = []
#         enemies = []
#
#         for other in state.players:
#             if other.car_id == player.car_id:
#                 continue
#
#             if other.team_num == player.team_num:
#                 team_obs = allies
#             else:
#                 team_obs = enemies
#
#             other_car = self._add_player_to_obs(team_obs, other, ball, inverted)
#
#             # Extra info
#             team_obs.extend([
#                 (other_car.position - player_car.position) / self.POS_STD,
#                 (other_car.linear_velocity - player_car.linear_velocity) / self.POS_STD
#             ])
#
#         obs.extend(allies)
#         obs.extend(enemies)
#         return np.concatenate(obs)
#
#     def _add_player_to_obs(self, obs: List, player: PlayerData, ball: PhysicsObject, inverted: bool):
#         if inverted:
#             player_car = player.inverted_car_data
#         else:
#             player_car = player.car_data
#
#         rel_pos = ball.position - player_car.position
#         rel_vel = ball.linear_velocity - player_car.linear_velocity
#
#         obs.extend([
#             rel_pos / self.POS_STD,
#             rel_vel / self.POS_STD,
#             player_car.position / self.POS_STD,
#             player_car.forward(),
#             player_car.up(),
#             player_car.linear_velocity / self.POS_STD,
#             player_car.angular_velocity / self.ANG_STD,
#             [player.boost_amount,
#              int(player.on_ground),
#              int(player.has_flip),
#              int(player.is_demoed)]])
#
#         return player_car

@njit(cache=True)
def _add_player_to_obs_jit(POS_STD: float, ANG_STD: float, obs: ndarray, i: int, ball_position: ndarray,
                           ball_linear_velocity: ndarray, position: ndarray, forward: ndarray,
                           up: ndarray, linear_velocity: ndarray, angular_velocity: ndarray,
                           boost_amount: float, on_ground: bool, has_flip: bool, is_demoed: bool,
                           player_pos: ndarray = None, player_lin_vel: ndarray = None):
    rel_pos = ball_position - position
    rel_vel = ball_linear_velocity - linear_velocity

    obs[i:i + 3] = rel_pos / POS_STD
    i += 3
    obs[i:i + 3] = rel_vel / POS_STD
    i += 3
    obs[i:i + 3] = position / POS_STD
    i += 3
    obs[i:i + 3] = forward
    i += 3
    obs[i:i + 3] = up
    i += 3
    obs[i:i + 3] = linear_velocity / POS_STD
    i += 3
    obs[i:i + 3] = angular_velocity / ANG_STD
    i += 3
    obs[i:i + 4] = [boost_amount,
                    int(on_ground),
                    int(has_flip),
                    int(is_demoed)]
    i += 4

    if player_pos is not None and player_lin_vel is not None:
        obs[i:i + 3] = (position - player_pos) / POS_STD
        i += 3
        obs[i:i + 3] = (linear_velocity - player_lin_vel) / POS_STD
        i += 3

    return i


class AdvancedObs(ObsBuilder):
    def __init__(self, expanding=False, team_size=1):
        super().__init__()
        self.POS_STD = 2300
        self.ANG_STD = pi
        self.expanding = expanding
        self.team_size = team_size
        self.obs_size = 51+25+(31*(self.team_size*2-1))

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: ndarray) -> Any:

        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball, pads = state.inverted_ball, state.inverted_boost_pads
        else:
            inverted = False
            ball, pads = state.ball, state.boost_pads

        pos_std, lin_std, ang_std = ball.position / self.POS_STD, ball.linear_velocity / self.POS_STD, \
                                    ball.angular_velocity / self.ANG_STD

        # TODO 1s for now, will work out if accurate later
        obs = zeros(self.obs_size)

        obs[0:3], obs[3:6], obs[6:9], obs[9:17], obs[17:51] = pos_std, lin_std, ang_std, previous_action, pads

        i = 51
        player_car, i = self._add_player_to_obs(obs, player, ball, inverted, i)
        # i = 76

        ally_count, enemy_count, i_enemies, i_allies = 0, 0, i+(31*(self.team_size-1)), i
        for other in state.players:
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                is_allies = True
                i = i_allies
            else:
                is_allies = False
                i = i_enemies

            other_car, i = self._add_player_to_obs(obs, other, ball, inverted, i, player_car)

            if is_allies:
                i_allies = i
            else:
                i_enemies = i

        if self.expanding:
            return expand_dims(obs, 0)

        return obs

    def _add_player_to_obs(self, obs: ndarray, car: PlayerData, ball: PhysicsObject,
                           inverted: bool, i: int, player: PhysicsObject = None):
        if inverted:
            player_car = car.inverted_car_data
        else:
            player_car = car.car_data
        if player is not None:
            i = _add_player_to_obs_jit(self.POS_STD, self.ANG_STD, obs, i, ball.position, ball.linear_velocity,
                                       player_car.position, player_car.forward(), player_car.up(),
                                       player_car.linear_velocity, player_car.angular_velocity,
                                       car.boost_amount, car.on_ground, car.has_flip, car.is_demoed,
                                       player.position, player.linear_velocity)
        else:
            i = _add_player_to_obs_jit(self.POS_STD, self.ANG_STD, obs, i, ball.position, ball.linear_velocity,
                                       player_car.position, player_car.forward(), player_car.up(),
                                       player_car.linear_velocity, player_car.angular_velocity,
                                       car.boost_amount, car.on_ground, car.has_flip, car.is_demoed)

        return player_car, i
