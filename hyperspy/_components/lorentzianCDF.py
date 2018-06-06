# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <http://www.gnu.org/licenses/>.

import math

import numpy as np

from hyperspy.component import Component


class LorentzianCDF(Component):

    """Arctan function component

    f(x) = A*(1/pi*arctan{(x-x0)/gamma}+1/2)

    +------------+-----------+
    | Parameter  | Attribute |
    +------------+-----------+
    +------------+-----------+
    |     A      |     A     |
    +------------+-----------+
    |   gamma    |   gamma   |
    +------------+-----------+
    |     x      |     x     |
    +------------+-----------+
    |     x0     |  centre   |
    +------------+-----------+

    """

    def __init__(self, A=1., gamma=1., centre=1.):
        Component.__init__(self, ['A', 'gamma', 'centre'])
        self.A.value = A
        self.A.grad = self.grad_A

        self.gamma.value = gamma
        self.gamma.grad = self.grad_gamma

        self.centre.value = centre
        self.centre.grad = self.grad_centre

        self.isbacgammaground = False
        self.isconvolved = False
        self._position = self.centre

    def function(self, x):
        A = self.A.value
        gamma = self.gamma.value
        centre = self.centre.value
        return A * (1/np.pi * np.arctan((x - centre) / gamma) + 1/2)

    def lorentzian(self, x):
        A = self.A.value
        gamma = self.gamma.value
        centre = self.centre.value
        return A / np.pi * (gamma / ((x - centre) ** 2 + gamma ** 2))

    def lorentzian_as_signal(self):
        import hs.model.components1d
        A = self.A.value
        gamma = self.gamma.value
        centre = self.centre.value
        return Lorentzian(A=A, gamma=gamma, center=center)

    def grad_A(self, x):
        gamma = self.gamma.value
        centre = self.centre.value
        return 1/np.pi * np.arctan((x - centre) / gamma) + 1/2

    def grad_gamma(self, x):
        A = self.A.value
        gamma = self.gamma.value
        centre = self.centre.value
        return (x - centre) / gamma * self.function(x)

    def grad_centre(self, x):
        return -self.function(x)
