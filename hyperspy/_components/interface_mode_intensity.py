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
from scipy.special import k0

from hyperspy.component import Component


class Interface_Mode_Intensity(Component):

    """Drude interface plasmon energy loss function for
    a beam in material 1 parallerl to an interface

    .. math::

       Energy loss function defined as:

       f(E) = \\Im\left( \frac{-2}{\epsilon_a (E)+\epsilon_b(E)}\right)

    +------------+-----------------+
    | Parameter  |    Attribute    |
    +------------+-----------------+
    | intensity  |        I1       |
    +------------+-----------------+
    |    E_p 1   |       Ep1       |
    +------------+-----------------+
    |delta_E_p 1 |       dE1       |
    +------------+-----------------+
    |    E_p 2   |       Ep2       |
    +------------+-----------------+
    |delta_E_p 2 |       dE2       |
    +------------+-----------------+

    Notes
    -----
    Wang, Z.L. “Valence Electron Excitations and Plasmon Oscillations in Thin
    Films, Surfaces, Interfaces and Small Particles.” Micron 27,
    no. 3–4 (June 1996): 265–99. doi:10.1016/0968-4328(96)00011-X.


    """

    def __init__(self, spectrum, A=1, C=1, x0=0, symetry='symetric'):
        self.symetry=symetry
        
        if symetry=='symetric':
            Component.__init__(self, ('A', 'C', 'x0'))
            self._position = x0
            
            #Define values
            self.x0.value = x0
            self.A.value = A #note A=I*Im(1/(eps1+eps2))
            self.C.value = C #C=omega/v
        elif symetry=='asymetric':
            Component.__init__(self, ('A_left', 'C_left', 'A_right', 'C_right', 'x0'))
            self._position = x0
            
            #Define values
            self.x0.value = x0
            self.A_left.value = A #note A=I*Im(1/(eps1+eps2))
            self.C_left.value = C #C=omega/v
            self.A_right.value = A #note A=I*Im(1/(eps1+eps2))
            self.C_right.value = C #C=omega/v
        
    def function(self, x):
        if self.symetry=='symetric':
            return np.where(
                    x != self.x0.value,
                    -1*self.A.value * k0(2 *self.C.value * abs(x - self.x0.value)),
                    0)
        elif self.symetry=='asymetric':
            return np.where(
                    x < self.x0.value,
                    -1*self.A_left.value * k0(2 *self.C_left.value * abs(x - self.x0.value)),
                    np.where(
                    x > self.x0.value,
                    -1*self.A_right.value * k0(2 *self.C_right.value * abs(x - self.x0.value)),
                    0))