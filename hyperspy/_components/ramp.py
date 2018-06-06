# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
from hyperspy.component import Component


class Ramp(Component):

    """The Ramp function

    f(x) = m*(x-x0)*Heaviside(x-x0)
    +------------+-----------+
    | Parameter  | Attribute |
    +------------+-----------+
    +------------+-----------+
    |   slope    |     m     |
    +------------+-----------+
    |   zero     |     x0    |
    +------------+-----------+
    """

    def __init__(self, m=1, x0=0):
        Component.__init__(self, ('m', 'x0'))
        self.m.value = m
        self.x0.value = x0
        self.isbackground = True
        self.convolved = False

        # Gradients
        self.m.grad = self.grad_m
        self.x0.grad = self.grad_x0

    def function(self, x):
        x = np.asanyarray(x)
        m = self.m.value
        x0 = self.x0.value
        return np.where(x < x0,
                        0,
                        m * (x - x0))

    def grad_m(self, x):
        x = np.asanyarray(x)
        m = self.m.value
        x0 = self.x0.value
        return np.where(x < x0,
                        0,
                        x - x0)

    def grad_x0(self, x):
        x = np.asanyarray(x)
        m = self.m.value
        x0 = self.x0.value
        return np.where(x < x0,
                        0,
                        np.where(x == x0,
                                 -0.5*m,
                                 -m)
                        )
