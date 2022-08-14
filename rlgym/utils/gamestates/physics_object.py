"""
A class to represent the state of a physics object from the game.
"""

# from numpy import zeros
from typing import Optional, List, Union

from numpy import ndarray, zeros, fromiter, ones

from rlgym.utils import math


# import numba
# from numba import float64, boolean
# from numba.experimental import jitclass
# from numba import types

# spec = [
#     ("position", types.optional(float64[::1])),
#     ("quaternion", types.optional(float64[::1])),
#     ("linear_velocity", types.optional(float64[::1])),
#     ("angular_velocity", types.optional(float64[::1])),
#     ("_euler_angles", float64[:]),
#     ("_rotation_mtx", float64[:, :]),
#     ("_has_computed_rot_mtx", boolean),
#     ("_has_computed_euler_angles", boolean)
# ]
#
#
# @jitclass(spec)
class PhysicsObject(object):
    def __init__(self, position=None, quaternion=None, linear_velocity=None, angular_velocity=None):
        # self.position: np.ndarray = position if position is not None else np.zeros(3)
        self.position: Union[ndarray, List] = position if position is not None else [0, 0, 0]

        # ones by default to prevent mathematical errors when converting quat to rot matrix on empty physics state
        # self.quaternion: np.ndarray = quaternion if quaternion is not None else np.ones(4)
        self.quaternion: Union[ndarray, List] = quaternion if quaternion is not None else [1, 1, 1, 1]

        # self.linear_velocity: np.ndarray = linear_velocity if linear_velocity is not None else np.zeros(3)
        self.linear_velocity: Union[ndarray, List] = linear_velocity if linear_velocity is not None else [1, 1, 1]
        # self.angular_velocity: np.ndarray = angular_velocity if angular_velocity is not None else np.zeros(3)
        self.angular_velocity: Union[ndarray, List] = angular_velocity if angular_velocity is not None else [0, 0, 0]
        # self._euler_angles: Optional[np.ndarray] = np.zeros(3)
        self._euler_angles: Optional[List] = [0, 0, 0]
        self._rotation_mtx: Optional[ndarray] = zeros((3, 3))
        # self._rotation_mtx: Optional[np.ndarray] = np.asarray([[0, 0, 0], [0, 0, 0]])
        self._has_computed_rot_mtx = False
        self._has_computed_euler_angles = False

    def decode_car_data(self, car_data):
        """
        Function to decode the physics state of a car from the game state array.
        :param car_data: Slice of game state array containing the car data to decode.
        """
        self.position = fromiter(car_data[:3], float)
        self.quaternion = fromiter(car_data[3:7], float)
        self.linear_velocity = fromiter(car_data[7:10], float)
        self.angular_velocity = fromiter(car_data[10:], float)

    def decode_ball_data(self, ball_data):
        """
        Function to decode the physics state of the ball from the game state array.
        :param ball_data: Slice of game state array containing the ball data to decode.
        """
        self.position = fromiter(ball_data[:3], float)
        self.linear_velocity = fromiter(ball_data[3:6], float)
        self.angular_velocity = fromiter(ball_data[6:9], float)

    def forward(self) -> ndarray:
        return self.rotation_mtx()[:, 0]

    def right(self) -> ndarray:
        return self.rotation_mtx()[:, 1]

    def left(self) -> ndarray:
        return self.rotation_mtx()[:, 1] * -1

    def up(self) -> ndarray:
        return self.rotation_mtx()[:, 2]

    def pitch(self) -> float:
        return self.euler_angles()[0]

    def yaw(self) -> float:
        return self.euler_angles()[1]

    def roll(self) -> float:
        return self.euler_angles()[2]

    # pitch, yaw, roll
    def euler_angles(self):
        if not self._has_computed_euler_angles:
            self._euler_angles = math.quat_to_euler(self.quaternion)
            self._has_computed_euler_angles = True

        return self._euler_angles

    def rotation_mtx(self):
        if not self._has_computed_rot_mtx:
            self._rotation_mtx = math.quat_to_rot_mtx_1d(self.quaternion)
            self._has_computed_rot_mtx = True

        return self._rotation_mtx

    def serialize(self):
        """
        Function to serialize all the values contained by this physics object into a single 1D list. This can be useful
        when constructing observations for a policy.
        :return: List containing the serialized data.
        """
        repr = []

        if self.position is not None:
            for arg in self.position:
                repr.append(arg)

        if self.quaternion is not None:
            for arg in self.quaternion:
                repr.append(arg)

        if self.linear_velocity is not None:
            for arg in self.linear_velocity:
                repr.append(arg)

        if self.angular_velocity is not None:
            for arg in self.angular_velocity:
                repr.append(arg)

        if self._euler_angles is not None:
            for arg in self._euler_angles:
                repr.append(arg)

        if self._rotation_mtx is not None:
            for arg in self._rotation_mtx.ravel():
                repr.append(arg)

        return repr


class FakePhysicsObject(object):
    def __init__(self, position=None, quaternion=None, linear_velocity=None, angular_velocity=None):
        self.position: ndarray = position if position is not None else zeros(3)
        # self.position: Union[ndarray, List] = position if position is not None else [0, 0, 0]

        # ones by default to prevent mathematical errors when converting quat to rot matrix on empty physics state
        self.quaternion: ndarray = quaternion if quaternion is not None else ones(4)
        # self.quaternion: Union[ndarray, List] = quaternion if quaternion is not None else [1, 1, 1, 1]

        self.linear_velocity: ndarray = linear_velocity if linear_velocity is not None else zeros(3)
        # self.linear_velocity: Union[ndarray, List] = linear_velocity if linear_velocity is not None else [1, 1, 1]
        self.angular_velocity: ndarray = angular_velocity if angular_velocity is not None else zeros(3)
        # self.angular_velocity: Union[ndarray, List] = angular_velocity if angular_velocity is not None else [0, 0, 0]
        self._euler_angles: Optional[ndarray] = zeros(3)
        # self._euler_angles: Optional[List] = [0, 0, 0]
        self._rotation_mtx: Optional[ndarray] = zeros((3, 3))
        # self._rotation_mtx: Optional[np.ndarray] = np.asarray([[0, 0, 0], [0, 0, 0]])
        self._has_computed_rot_mtx = False
        self._has_computed_euler_angles = False

    def decode_car_data(self, car_data):
        """
        Function to decode the physics state of a car from the game state array.
        :param car_data: Slice of game state array containing the car data to decode.
        """
        self.position = fromiter(car_data[:3], float)
        self.quaternion = fromiter(car_data[3:7], float)
        self.linear_velocity = fromiter(car_data[7:10], float)
        self.angular_velocity = fromiter(car_data[10:], float)

    def decode_ball_data(self, ball_data):
        """
        Function to decode the physics state of the ball from the game state array.
        :param ball_data: Slice of game state array containing the ball data to decode.
        """
        self.position = fromiter(ball_data[:3], float)
        self.linear_velocity = fromiter(ball_data[3:6], float)
        self.angular_velocity = fromiter(ball_data[6:9], float)

    def forward(self) -> ndarray:
        return self.rotation_mtx()[:, 0]

    def right(self) -> ndarray:
        return self.rotation_mtx()[:, 1]

    def left(self) -> ndarray:
        return self.rotation_mtx()[:, 1] * -1

    def up(self) -> ndarray:
        return self.rotation_mtx()[:, 2]

    def pitch(self) -> float:
        return self.euler_angles()[0]

    def yaw(self) -> float:
        return self.euler_angles()[1]

    def roll(self) -> float:
        return self.euler_angles()[2]

    # pitch, yaw, roll
    def euler_angles(self):
        if not self._has_computed_euler_angles:
            self._euler_angles = math.quat_to_euler(self.quaternion)
            self._has_computed_euler_angles = True

        return self._euler_angles

    def rotation_mtx(self):
        if not self._has_computed_rot_mtx:
            self._rotation_mtx = math.quat_to_rot_mtx_1d(self.quaternion)
            self._has_computed_rot_mtx = True

        return self._rotation_mtx

    def serialize(self):
        """
        Function to serialize all the values contained by this physics object into a single 1D list. This can be useful
        when constructing observations for a policy.
        :return: List containing the serialized data.
        """
        repr = []

        if self.position is not None:
            for arg in self.position:
                repr.append(arg)

        if self.quaternion is not None:
            for arg in self.quaternion:
                repr.append(arg)

        if self.linear_velocity is not None:
            for arg in self.linear_velocity:
                repr.append(arg)

        if self.angular_velocity is not None:
            for arg in self.angular_velocity:
                repr.append(arg)

        if self._euler_angles is not None:
            for arg in self._euler_angles:
                repr.append(arg)

        if self._rotation_mtx is not None:
            for arg in self._rotation_mtx.ravel():
                repr.append(arg)

        return repr
