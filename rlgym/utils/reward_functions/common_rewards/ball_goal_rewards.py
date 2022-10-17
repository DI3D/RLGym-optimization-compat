import numpy as np
import math as math_py

from rlgym.utils import RewardFunction, math
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, ORANGE_GOAL_BACK, \
    BLUE_GOAL_BACK, BALL_MAX_SPEED, BACK_WALL_Y, BALL_RADIUS, BACK_NET_Y
from rlgym.utils.gamestates import GameState, PlayerData


class LiuDistanceBallToGoalReward(RewardFunction):
    def __init__(self, own_goal=False):
        super().__init__()
        self.own_goal = own_goal
        self.partial = (BACK_NET_Y - BACK_WALL_Y + BALL_RADIUS)

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM and not self.own_goal \
                or player.team_num == ORANGE_TEAM and self.own_goal:
            objective = ORANGE_GOAL_BACK
        else:
            objective = BLUE_GOAL_BACK

        # Compensate for moving objective to back of net
        # dist = np.linalg.norm([i - j for i, j in zip(state.ball.position - objective)]) - (
        # BACK_NET_Y - BACK_WALL_Y + BALL_RADIUS)
        # dist = np.linalg.norm([i - j for i, j in zip([i - j for i, j in zip(state.ball.position, objective)])]) - (
        #             BACK_NET_Y - BACK_WALL_Y + BALL_RADIUS)
        dist = math.norm_1d([i - j for i, j in zip([i - j for i, j in zip(state.ball.position, objective)])]) \
               - self.partial
        # dist = np.linalg.norm(state.ball.position - objective) - (BACK_NET_Y - BACK_WALL_Y + BALL_RADIUS)
        return math_py.exp(-0.5 * dist / BALL_MAX_SPEED)  # Inspired by https://arxiv.org/abs/2105.12196


class VelocityBallToGoalReward(RewardFunction):
    def __init__(self, own_goal=False, use_scalar_projection=False):
        super().__init__()
        self.own_goal = own_goal
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM and not self.own_goal \
                or player.team_num == ORANGE_TEAM and self.own_goal:
            objective = ORANGE_GOAL_BACK
        else:
            objective = BLUE_GOAL_BACK

        # vel = state.ball.linear_velocity
        # pos_diff = objective - state.ball.position
        pos_diff = [i - j for i, j in zip(objective, state.ball.position)]
        # pos_diff_arr = np.fromiter(pos_diff, dtype=np.float64, count=len(pos_diff))
        # pos_diff_normed = np.linalg.norm(pos_diff_arr)
        # pos_diff_normed = math.norm_1d(pos_diff)
        if self.use_scalar_projection:
            # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
            # Max value should be max_speed / ball_radius = 2300 / 94 = 24.5
            # Used to guide the agent towards the ball
            # inv_t = math.scalar_projection_1d(list(state.ball.linear_velocity), pos_diff)
            return math.scalar_projection_1d(list(state.ball.linear_velocity), pos_diff)
        else:
            # Regular component velocity
            # norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            pos_diff_normed = math.norm_1d(pos_diff)
            norm_pos_diff = [i / pos_diff_normed for i in pos_diff]
            # norm_vel = vel / BALL_MAX_SPEED
            norm_vel = [i / BALL_MAX_SPEED for i in state.ball.linear_velocity]
            # return float(np.dot(norm_pos_diff, norm_vel))
            return sum([i * j for i, j in zip(norm_pos_diff, norm_vel)])


class BallYCoordinateReward(RewardFunction):
    def __init__(self, exponent=1):
        # Exponent should be odd so that negative y -> negative reward
        self.exponent = exponent

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM:
            return (state.ball.position[1] / (BACK_WALL_Y + BALL_RADIUS)) ** self.exponent
        else:
            return (state.inverted_ball.position[1] / (BACK_WALL_Y + BALL_RADIUS)) ** self.exponent
