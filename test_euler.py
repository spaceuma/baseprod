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

"""Tests for the conversion between Euler and Quaternion."""

___author___ = "Levin Gerdes"


import unittest

from numpy import pi

from euler import (
    Euler,
    Quaternion,
    euler_to_quaternion,
    quaternion_to_euler,
    wrap_angle,
    wrap_euler,
)


class TestWrapAngle(unittest.TestCase):
    def test_no_change(self):
        angles = [-pi, -pi / 2, 0, pi / 2, pi - 1e-5]
        for x in angles:
            self.assertAlmostEqual(x, wrap_angle(x))

    def test_positive(self):
        self.assertAlmostEqual(wrap_angle(pi), -pi)
        self.assertAlmostEqual(wrap_angle(2 * pi), 0)
        self.assertAlmostEqual(wrap_angle(1.5 * pi), -0.5 * pi)

    def test_negative(self):
        self.assertAlmostEqual(wrap_angle(-pi - 1), pi - 1)
        self.assertAlmostEqual(wrap_angle(-2 * pi), 0)
        self.assertAlmostEqual(wrap_angle(-1.5 * pi), 0.5 * pi)


class TestWrapEuler(unittest.TestCase):
    def test_no_change(self):
        eulers = [Euler(-pi, -pi, -pi), Euler(0, -1, -2), Euler(1, 2, pi - 1e-5)]
        for e in eulers:
            e2 = wrap_euler(e)
            self.assertAlmostEqual(e.x, e2.x)
            self.assertAlmostEqual(e.y, e2.y)
            self.assertAlmostEqual(e.z, e2.z)

    def test_positive(self):
        e = Euler(pi, 2 * pi, 3 * pi)
        e2 = wrap_euler(e)
        self.assertAlmostEqual(e2.x, -pi)
        self.assertAlmostEqual(e2.y, 0)
        self.assertAlmostEqual(e2.z, -pi)

    def test_negative(self):
        e = Euler(-pi - 1, -2 * pi, -3 * pi)
        e2 = wrap_euler(e)
        self.assertAlmostEqual(e2.x, pi - 1)
        self.assertAlmostEqual(e2.y, 0)
        self.assertAlmostEqual(e2.z, -pi)


class TestEulerToQuaternion(unittest.TestCase):
    def test_x_only(self):
        e = Euler(1, 0, 0)
        q = euler_to_quaternion(e)
        self.assertAlmostEqual(q.x, 0.4794255)
        self.assertAlmostEqual(q.y, 0)
        self.assertAlmostEqual(q.z, 0)
        self.assertAlmostEqual(q.w, 0.8775826)

    def test_y_only(self):
        e = Euler(0, 1, 0)
        q = euler_to_quaternion(e)
        self.assertAlmostEqual(q.x, 0)
        self.assertAlmostEqual(q.y, 0.4794255)
        self.assertAlmostEqual(q.z, 0)
        self.assertAlmostEqual(q.w, 0.8775826)

    def test_z_only(self):
        e = Euler(0, 0, 1)
        q = euler_to_quaternion(e)
        self.assertAlmostEqual(q.x, 0)
        self.assertAlmostEqual(q.y, 0)
        self.assertAlmostEqual(q.z, 0.4794255)
        self.assertAlmostEqual(q.w, 0.8775826)

    def test_all(self):
        e = Euler(2, 1.5, 1)
        q = euler_to_quaternion(e)
        self.assertAlmostEqual(q.x, 0.363755)
        self.assertAlmostEqual(q.y, 0.6183856)
        self.assertAlmostEqual(q.z, -0.3138303)
        self.assertAlmostEqual(q.w, 0.621926)

    def test_x_negative(self):
        e = Euler(-1, 1.5, 1)
        q = euler_to_quaternion(e)
        self.assertAlmostEqual(q.x, -0.5946371)
        self.assertAlmostEqual(q.y, 0.356787)
        self.assertAlmostEqual(q.z, 0.5946371)
        self.assertAlmostEqual(q.w, 0.4068371)


class TestQuaternionToEuler(unittest.TestCase):
    def test_x_component(self):
        q = Quaternion(1, 0, 0, 1)
        e = quaternion_to_euler(q)
        self.assertAlmostEqual(e.x, 1.5707963)
        self.assertAlmostEqual(e.y, 0)
        self.assertAlmostEqual(e.z, 0)

    def test_y_component(self):
        q = Quaternion(0, 1, 0, 1)
        e = quaternion_to_euler(q)
        self.assertAlmostEqual(e.x, 0)
        self.assertAlmostEqual(e.y, 1.5707963)
        self.assertAlmostEqual(e.z, 0)

    def test_z_component(self):
        q = Quaternion(0, 0, 1, 1)
        e = quaternion_to_euler(q)
        self.assertAlmostEqual(e.x, 0)
        self.assertAlmostEqual(e.y, 0)
        self.assertAlmostEqual(e.z, 1.5707963)


class TestBackAndForth(unittest.TestCase):
    def test_q_e_q(self):
        q = Quaternion(1, 0, 0, 0)
        q2 = euler_to_quaternion(quaternion_to_euler(q))
        self.assertAlmostEqual(q.x, q2.x)
        self.assertAlmostEqual(q.y, q2.y)
        self.assertAlmostEqual(q.z, q2.z)
        self.assertAlmostEqual(q.w, q2.w)

    def test_e_q_e(self):
        e = Euler(1, 1.2, -1.2)
        e2 = quaternion_to_euler(euler_to_quaternion(e))
        self.assertAlmostEqual(e.x, e2.x)
        self.assertAlmostEqual(e.y, e2.y)
        self.assertAlmostEqual(e.z, e2.z)


if __name__ == "__main__":
    unittest.main()
