# MIT License
#
# Copyright (c) 2024 Space Robotics Lab at UMA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Convert Quaternion to Euler in ZYX sequence."""

__author__ = "Levin Gerdes"


from dataclasses import dataclass

from numpy import pi
from scipy.spatial.transform import Rotation as R  # type: ignore


@dataclass
class Euler:
    x: float
    y: float
    z: float


@dataclass
class Quaternion:
    x: float
    y: float
    z: float
    w: float


def wrap_angle(x: float) -> float:
    """
    Wraps an angle to [-pi, pi)
    """
    return (x + pi) % (2 * pi) - pi


def wrap_euler(e: Euler) -> Euler:
    """
    Wraps each component to [-pi, pi)
    """
    e.x = wrap_angle(e.x)
    e.y = wrap_angle(e.y)
    e.z = wrap_angle(e.z)
    return e


def quaternion_to_euler(q: Quaternion) -> Euler:
    """
    Converts Quaternion to Euler in ZYX rotation order
    """
    r = R.from_quat([q.x, q.y, q.z, q.w])
    re = r.as_euler("ZYX")
    e = Euler(z=re[0], y=re[1], x=re[2])
    return e


def euler_to_quaternion(e: Euler) -> Quaternion:
    """
    Euler in radians and ZYX rotation order to Quaternion
    """
    r = R.from_euler("ZYX", [e.z, e.y, e.x], degrees=False)
    rq = r.as_quat()
    q = Quaternion(x=rq[0], y=rq[1], z=rq[2], w=rq[3])
    return q
