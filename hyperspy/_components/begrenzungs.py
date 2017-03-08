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


class Begrenzungs(Component):

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

    def __init__(self, spectrum, A=1, B=1, C=1, x0=0, side='Both'):
        Component.__init__(self, ('A', 'B', 'C', 'x0'))
        self._position = x0
        self._side =side
        
        #Define values
        self.x0.value = x0
        self.A.value = A #note A=I*Im(1/eps)
        self.B.value = B #note B=kym*v/omega
        self.C.value = C #C=omega/v
        
        #Get scope meta
#        beta = spectrum.metadata.Acquisition_instrument.TEM.Detector.EELS.collection_angle/1000 #[mrad]
#        E0 = spectrum.metadata.Acquisition_instrument.TEM.beam_energy #[keV]
#        hbar=6.582E-16#[eV s]
#        c=3E8#[m/s]
#        m0c2=511.#[keV]
#        k_0 = np.sqrt(2*m0c2*E0/c**2 * (1 + E0/(2*m0c2)))/hbar #[1/m]
#        self.kym = beta*k_0 #[1/m]

        
    def function(self, x):
        #kym = self.kym
        if self._side == 'Both':
            return np.where(
                    x != self.x0.value,
                    self.A.value * (np.log(1/self.B.value) - k0(2 *self.C.value * (self.x0.value - x))),
                    0)
        elif self._side == 'Left':
            return np.where(
                    x < self.x0.value,
                    self.A.value * (np.log(1/self.B.value) - k0(2 * self.C.value * (self.x0.value - x))),
                    0)
        elif self._side == 'Right':
            return np.where(
                    x > self.x0.value,
                    self.A.value * (np.log(1/self.B.value) - k0(2 * self.C.value * (self.x0.value - x))),
                    0)