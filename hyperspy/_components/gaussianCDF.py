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
from scipy.special import erf

from hyperspy.component import Component

sqrt2pi = np.sqrt(2 * np.pi)


class GaussianCDF(Component):

    """Cumlative distribution function for a gaussian.

    Attributes
    ----------
    A : float
    sigma : float
    origin : float
    """

    def __init__(self):
        Component.__init__(self, ['A', 'sigma', 'origin'])

        # Boundaries
        self.A.bmin = 0.
        self.A.bmax = None

        self.sigma.bmin = None
        self.sigma.bmax = None

        self.isbackground = False
        self.convolved = True

        # Gradients
        self.A.grad = self.grad_A
        self.sigma.grad = self.grad_sigma
        self.origin.grad = self.grad_origin
        self._position = self.origin

    def function(self, x):
        A = self.A.value
        sigma = self.sigma.value
        origin = self.origin.value
        return A/2 * (1+ erf((x - origin) / np.sqrt(2) / sigma))

    def gaussian(self, x):
        A = self.A.value
        sigma = self.sigma.value
        origin = self.origin.value
        return (A / (sqrt2pi * sigma) *
            np.exp(-((x -origin)/sigma)**2 / 2))

    def gaussian_as_signal(self):
        import hs.model.components1d
        A = self.A.value
        sigma = self.sigma.value
        origin = self.origin.value
        return Gaussian(A=A, sigma=sigma, origin=origin)

    def grad_A(self, x):
        sigma = self.sigma.value
        origin = self.origin.value
        return 1/2 * (1 + erf((x - origin) / np.sqrt(2) / sigma))

    def grad_sigma(self, x):
        sigma = self.sigma.value
        origin = self.origin.value
        return -(x - origin)/sigma * self.gaussian(x)

    def grad_origin(self, x):
        return self.gaussian(x)
