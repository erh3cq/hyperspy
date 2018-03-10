# -*- coding: utf-8 -*-
# Copyright 2007-2015 The HyperSpy developers
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

# The EMD format is a hdf5 standard proposed at Lawrence Berkeley
# National Lab (see http://emdatasets.com/ for more information).
# NOT to be confused with the FEI EMD format which was developed later.


import os.path
from os import remove
import shutil
import tempfile
from numpy.testing import assert_allclose
import numpy as np
import h5py
from dateutil import tz
from datetime import datetime

from hyperspy.io import load
from hyperspy.signals import BaseSignal, Signal2D, Signal1D, EDSTEMSpectrum
from hyperspy.misc.test_utils import assert_deep_almost_equal


my_path = os.path.dirname(__file__)

# Reference data:
data_signal = np.arange(27).reshape((3, 3, 3))
data_image = np.arange(9).reshape((3, 3))
data_spectrum = np.arange(3)
data_save = np.arange(24).reshape((2, 3, 4))
sig_metadata = {'a': 1, 'b': 2}
user = {'name': 'John Doe', 'institution': 'TestUniversity',
        'department': 'Microscopy', 'email': 'johndoe@web.de'}
microscope = {'name': 'Titan', 'voltage': '300kV'}
sample = {'material': 'TiO2', 'preparation': 'FIB'}
comments = {'comment': 'Test'}
test_title = 'This is a test!'


def test_signal_3d_loading():
    signal = load(os.path.join(my_path, 'emd_files', 'example_signal.emd'))
    np.testing.assert_equal(signal.data, data_signal)
    assert isinstance(signal, BaseSignal)


def test_image_2d_loading():
    signal = load(os.path.join(my_path, 'emd_files', 'example_image.emd'))
    np.testing.assert_equal(signal.data, data_image)
    assert isinstance(signal, Signal2D)


def test_spectrum_1d_loading():
    signal = load(os.path.join(my_path, 'emd_files', 'example_spectrum.emd'))
    np.testing.assert_equal(signal.data, data_spectrum)
    assert isinstance(signal, Signal1D)


def test_metadata():
    signal = load(os.path.join(my_path, 'emd_files', 'example_metadata.emd'))
    np.testing.assert_equal(signal.data, data_image)
    np.testing.assert_equal(signal.metadata.General.title, test_title)
    np.testing.assert_equal(signal.metadata.General.user.as_dictionary(), user)
    np.testing.assert_equal(
        signal.metadata.General.microscope.as_dictionary(),
        microscope)
    np.testing.assert_equal(
        signal.metadata.General.sample.as_dictionary(), sample)
    np.testing.assert_equal(
        signal.metadata.General.comments.as_dictionary(),
        comments)
    for key, ref_value in sig_metadata.items():
        np.testing.assert_equal(
            signal.metadata.Signal.as_dictionary().get(key), ref_value)
    assert isinstance(signal, Signal2D)


def test_metadata_with_bytes_string():
    filename = os.path.join(
        my_path, 'emd_files', 'example_bytes_string_metadata.emd')
    f = h5py.File(filename, 'r')
    dim1 = f['test_group']['data_group']['dim1']
    dim1_name = dim1.attrs['name']
    dim1_units = dim1.attrs['units']
    f.close()
    assert isinstance(dim1_name, np.bytes_)
    assert isinstance(dim1_units, np.bytes_)
    signal = load(os.path.join(my_path, 'emd_files', filename))


def test_data_numpy_object_dtype():
    filename = os.path.join(
        my_path, 'emd_files', 'example_object_dtype_data.emd')
    signal = load(filename)
    assert len(signal) == 0


def test_data_axis_length_1():
    filename = os.path.join(
        my_path, 'emd_files', 'example_axis_len_1.emd')
    signal = load(filename)
    assert signal.data.shape == (5, 1, 5)


class TestMinimalSave():

    def test_minimal_save(self):
        self.signal = Signal1D([0, 1])
        with tempfile.TemporaryDirectory() as tmp:
            self.signal.save(os.path.join(tmp, 'testfile.emd'))


class TestCaseSaveAndRead():

    def test_save_and_read(self):
        signal_ref = BaseSignal(data_save)
        signal_ref.metadata.General.title = test_title
        signal_ref.axes_manager[0].name = 'x'
        signal_ref.axes_manager[1].name = 'y'
        signal_ref.axes_manager[2].name = 'z'
        signal_ref.axes_manager[0].scale = 2
        signal_ref.axes_manager[1].scale = 3
        signal_ref.axes_manager[2].scale = 4
        signal_ref.axes_manager[0].offset = 10
        signal_ref.axes_manager[1].offset = 20
        signal_ref.axes_manager[2].offset = 30
        signal_ref.axes_manager[0].units = 'nmx'
        signal_ref.axes_manager[1].units = 'nmy'
        signal_ref.axes_manager[2].units = 'nmz'
        signal_ref.save(os.path.join(my_path, 'emd_files', 'example_temp.emd'), overwrite=True,
                        signal_metadata=sig_metadata, user=user, microscope=microscope,
                        sample=sample, comments=comments)
        signal = load(os.path.join(my_path, 'emd_files', 'example_temp.emd'))
        np.testing.assert_equal(signal.data, signal_ref.data)
        np.testing.assert_equal(signal.axes_manager[0].name, 'x')
        np.testing.assert_equal(signal.axes_manager[1].name, 'y')
        np.testing.assert_equal(signal.axes_manager[2].name, 'z')
        np.testing.assert_equal(signal.axes_manager[0].scale, 2)
        np.testing.assert_equal(signal.axes_manager[1].scale, 3)
        np.testing.assert_equal(signal.axes_manager[2].scale, 4)
        np.testing.assert_equal(signal.axes_manager[0].offset, 10)
        np.testing.assert_equal(signal.axes_manager[1].offset, 20)
        np.testing.assert_equal(signal.axes_manager[2].offset, 30)
        np.testing.assert_equal(signal.axes_manager[0].units, 'nmx')
        np.testing.assert_equal(signal.axes_manager[1].units, 'nmy')
        np.testing.assert_equal(signal.axes_manager[2].units, 'nmz')
        np.testing.assert_equal(signal.metadata.General.title, test_title)
        np.testing.assert_equal(
            signal.metadata.General.user.as_dictionary(), user)
        np.testing.assert_equal(
            signal.metadata.General.microscope.as_dictionary(),
            microscope)
        np.testing.assert_equal(
            signal.metadata.General.sample.as_dictionary(), sample)
        np.testing.assert_equal(
            signal.metadata.General.comments.as_dictionary(), comments)
        for key, ref_value in sig_metadata.items():
            np.testing.assert_equal(
                signal.metadata.Signal.as_dictionary().get(key), ref_value)
        assert isinstance(signal, BaseSignal)

    def teardown_method(self, method):
        remove(os.path.join(my_path, 'emd_files', 'example_temp.emd'))


class TestFeiEMD():

    fei_files_path = os.path.join(my_path, "emd_files", "fei_emd_files")

    @classmethod
    def setup_class(cls):
        import zipfile
        zipf = os.path.join(my_path, "emd_files", "fei_emd_files.zip")
        with zipfile.ZipFile(zipf, 'r') as zipped:
            zipped.extractall(cls.fei_files_path)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.fei_files_path)

    def test_fei_emd_image(self):
        stage = {'tilt_alpha': '0.00',
                 'tilt_beta': '0.00',
                 'x': '-0.000',
                 'y': '0.000',
                 'z': '0.000'}
        md = {'Acquisition_instrument': {'TEM': {'beam_energy': 200.0,
                                                 'camera_length': 98.0,
                                                 'magnification': 40000.0,
                                                 'microscope': 'Talos',
                                                 'Stage': stage}},
              'General': {'original_filename': 'fei_emd_image.emd',
                          'date': '2017-03-06',
                          'time': '09:56:41',
                          'time_zone': 'BST',
                          'title': 'HAADF'},
              'Signal': {'binned': False, 'signal_type': 'image'},
              '_HyperSpy': {'Folding': {'original_axes_manager': None,
                                        'original_shape': None,
                                        'signal_unfolded': False,
                                        'unfolded': False}}}

        # Update time and time_zone to local ones
        md['General']['time_zone'] = tz.tzlocal().tzname(datetime.today())
        dt = datetime.fromtimestamp(1488794201, tz=tz.tzutc())
        date, time = dt.astimezone(
            tz.tzlocal()).isoformat().split('+')[0].split('T')
        md['General']['date'] = date
        md['General']['time'] = time

        signal = load(os.path.join(self.fei_files_path, 'fei_emd_image.emd'))
        fei_image = np.load(os.path.join(self.fei_files_path,
                                         'fei_emd_image.npy'))
        assert signal.axes_manager[0].name == 'x'
        assert signal.axes_manager[0].units == 'um'
        assert_allclose(signal.axes_manager[0].scale, 0.005302, atol=1E-5)
        assert signal.axes_manager[1].name == 'y'
        assert signal.axes_manager[1].units == 'um'
        assert_allclose(signal.axes_manager[1].scale, 0.005302, atol=1E-5)
        assert_allclose(signal.data, fei_image)
        assert_deep_almost_equal(signal.metadata.as_dictionary(), md)
        assert isinstance(signal, Signal2D)

    def test_fei_emd_spectrum(self):
        signal = load(os.path.join(
            self.fei_files_path, 'fei_emd_spectrum.emd'))
        fei_spectrum = np.load(os.path.join(self.fei_files_path,
                                            'fei_emd_spectrum.npy'))
        np.testing.assert_equal(signal.data, fei_spectrum)
        assert isinstance(signal, Signal1D)

    def test_fei_emd_si(self):
        signal = load(os.path.join(self.fei_files_path, 'fei_emd_si.emd'))
        fei_si = np.load(os.path.join(self.fei_files_path, 'fei_emd_si.npy'))
        np.testing.assert_equal(signal[1].data, fei_si)
        assert isinstance(signal[1], Signal1D)

    def test_fei_emd_si_non_square_10frames(self):
        s = load(os.path.join(self.fei_files_path,
                              'fei_SI_SuperX-HAADF_10frames_10x50.emd'))
        signal = s[1]
        assert isinstance(signal, EDSTEMSpectrum)
        assert signal.axes_manager[0].name == 'x'
        assert signal.axes_manager[0].size == 10
        assert signal.axes_manager[0].units == 'nm'
        assert_allclose(signal.axes_manager[0].scale, 1.234009, atol=1E-5)
        assert signal.axes_manager[1].name == 'y'
        assert signal.axes_manager[1].size == 50
        assert signal.axes_manager[1].units == 'nm'
        assert_allclose(signal.axes_manager[1].scale, 1.234009, atol=1E-5)
        assert signal.axes_manager[2].name == 'X-ray energy'
        assert signal.axes_manager[2].size == 4096
        assert signal.axes_manager[2].units == 'keV'
        assert_allclose(signal.axes_manager[2].scale, 0.005, atol=1E-5)
        assert signal.metadata.Acquisition_instrument.TEM.Detector.EDS.number_of_frames == 10

        signal0 = s[0]
        assert isinstance(signal0, Signal2D)
        assert signal0.axes_manager[0].name == 'x'
        assert signal0.axes_manager[0].size == 10
        assert signal0.axes_manager[0].units == 'nm'
        assert_allclose(signal0.axes_manager[0].scale, 1.234009, atol=1E-5)
        assert signal0.axes_manager[1].name == 'y'
        assert signal0.axes_manager[1].size == 50
        assert signal0.axes_manager[1].units == 'nm'

        s = load(os.path.join(self.fei_files_path,
                              'fei_SI_SuperX-HAADF_10frames_10x50.emd'),
                 sum_frames=False,
                 SI_dtype=np.uint8,
                 rebin_energy=256)
        signal = s[1]
        assert isinstance(signal, EDSTEMSpectrum)
        assert signal.axes_manager.navigation_shape == (10, 50, 10)
        assert signal.axes_manager[0].name == 'x'
        assert signal.axes_manager[0].size == 10
        assert signal.axes_manager[0].units == 'nm'
        assert_allclose(signal.axes_manager[0].scale, 1.234009, atol=1E-5)
        assert signal.axes_manager[1].name == 'y'
        assert signal.axes_manager[1].size == 50
        assert signal.axes_manager[1].units == 'nm'
        assert_allclose(signal.axes_manager[1].scale, 1.234009, atol=1E-5)
        assert signal.axes_manager[2].name == 'Time'
        assert signal.axes_manager[2].size == 10
        assert signal.axes_manager[2].units == 's'
        assert_allclose(signal.axes_manager[2].scale, 0.76800, atol=1E-5)
        assert signal.axes_manager[3].name == 'X-ray energy'
        assert signal.axes_manager[3].size == 16
        assert signal.axes_manager[3].units == 'keV'
        assert_allclose(signal.axes_manager[3].scale, 1.28, atol=1E-5)
        assert signal.metadata.Acquisition_instrument.TEM.Detector.EDS.number_of_frames == 10

        s = load(os.path.join(self.fei_files_path,
                              'fei_SI_SuperX-HAADF_10frames_10x50.emd'),
                 sum_frames=False,
                 last_frame=5,
                 SI_dtype=np.uint8,
                 rebin_energy=256)
        signal = s[1]
        assert isinstance(signal, EDSTEMSpectrum)
        assert signal.axes_manager.navigation_shape == (10, 50, 5)
        assert signal.axes_manager[0].name == 'x'
        assert signal.axes_manager[0].size == 10
        assert signal.axes_manager[0].units == 'nm'
        assert_allclose(signal.axes_manager[0].scale, 1.234009, atol=1E-5)
        assert signal.axes_manager[1].name == 'y'
        assert signal.axes_manager[1].size == 50
        assert signal.axes_manager[1].units == 'nm'
        assert_allclose(signal.axes_manager[1].scale, 1.234009, atol=1E-5)
        assert signal.axes_manager[2].name == 'Time'
        assert signal.axes_manager[2].size == 5
        assert signal.axes_manager[2].units == 's'
        assert_allclose(signal.axes_manager[2].scale, 0.76800, atol=1E-5)
        assert signal.axes_manager[3].name == 'X-ray energy'
        assert signal.axes_manager[3].size == 16
        assert signal.axes_manager[3].units == 'keV'
        assert_allclose(signal.axes_manager[3].scale, 1.28, atol=1E-5)
        assert signal.metadata.Acquisition_instrument.TEM.Detector.EDS.number_of_frames == 5

        s = load(os.path.join(self.fei_files_path,
                              'fei_SI_SuperX-HAADF_10frames_10x50.emd'),
                 sum_frames=False,
                 first_frame=4,
                 SI_dtype=np.uint8,
                 rebin_energy=256)
        signal = s[1]
        assert isinstance(signal, EDSTEMSpectrum)
        assert signal.axes_manager.navigation_shape == (10, 50, 6)
        assert signal.axes_manager[0].name == 'x'
        assert signal.axes_manager[0].size == 10
        assert signal.axes_manager[0].units == 'nm'
        assert_allclose(signal.axes_manager[0].scale, 1.234009, atol=1E-5)
        assert signal.axes_manager[1].name == 'y'
        assert signal.axes_manager[1].size == 50
        assert signal.axes_manager[1].units == 'nm'
        assert_allclose(signal.axes_manager[1].scale, 1.234009, atol=1E-5)
        assert signal.axes_manager[2].name == 'Time'
        assert signal.axes_manager[2].size == 6
        assert signal.axes_manager[2].units == 's'
        assert_allclose(signal.axes_manager[2].scale, 0.76800, atol=1E-5)
        assert signal.axes_manager[3].name == 'X-ray energy'
        assert signal.axes_manager[3].size == 16
        assert signal.axes_manager[3].units == 'keV'
        assert_allclose(signal.axes_manager[3].scale, 1.28, atol=1E-5)
        assert signal.metadata.Acquisition_instrument.TEM.Detector.EDS.number_of_frames == 6

    def test_fei_emd_si_non_square_20frames(self):
        s = load(os.path.join(self.fei_files_path,
                              'fei_SI_SuperX-HAADF_20frames_10x50.emd'))
        signal = s[1]
        assert isinstance(signal, EDSTEMSpectrum)
        assert signal.axes_manager[0].name == 'x'
        assert signal.axes_manager[0].size == 10
        assert signal.axes_manager[0].units == 'nm'
        assert_allclose(signal.axes_manager[0].scale, 1.234009, atol=1E-5)
        assert signal.axes_manager[1].name == 'y'
        assert signal.axes_manager[1].size == 50
        assert signal.axes_manager[1].units == 'nm'
        assert_allclose(signal.axes_manager[1].scale, 1.234009, atol=1E-5)
        assert signal.axes_manager[2].name == 'X-ray energy'
        assert signal.axes_manager[2].size == 4096
        assert signal.axes_manager[2].units == 'keV'
        assert_allclose(signal.axes_manager[2].scale, 0.005, atol=1E-5)
        assert signal.metadata.Acquisition_instrument.TEM.Detector.EDS.number_of_frames == 20

    def test_fei_emd_si_non_square_20frames_2eV(self):
        s = load(os.path.join(self.fei_files_path,
                              'fei_SI_SuperX-HAADF_20frames_10x50_2ev.emd'))
        signal = s[1]
        assert isinstance(signal, EDSTEMSpectrum)
        assert signal.axes_manager[0].name == 'x'
        assert signal.axes_manager[0].size == 10
        assert signal.axes_manager[0].units == 'nm'
        assert_allclose(signal.axes_manager[0].scale, 1.234009, atol=1E-5)
        assert signal.axes_manager[1].name == 'y'
        assert signal.axes_manager[1].size == 50
        assert signal.axes_manager[1].units == 'nm'
        assert_allclose(signal.axes_manager[1].scale, 1.234009, atol=1E-5)
        assert signal.axes_manager[2].name == 'X-ray energy'
        assert signal.axes_manager[2].size == 4096
        assert signal.axes_manager[2].units == 'keV'
        assert_allclose(signal.axes_manager[2].scale, 0.002, atol=1E-5)
        assert signal.metadata.Acquisition_instrument.TEM.Detector.EDS.number_of_frames == 20

    def test_fei_emd_si_frame_range(self):
        signal = load(os.path.join(self.fei_files_path, 'fei_emd_si.emd'),
                      first_frame=2, last_frame=4)
        fei_si = np.load(os.path.join(self.fei_files_path,
                                      'fei_emd_si_frame.npy'))
        np.testing.assert_equal(signal[1].data, fei_si)
        assert isinstance(signal[1], Signal1D)
        md = signal[1].metadata
        assert md['Acquisition_instrument']['TEM']['Detector']['EDS']['number_of_frames'] == 2

    def time_loading_frame(self):
        # Run this function to check the loading time when loading EDS data
        import time
        frame_number = 100
        point_measurement = 15
        frame_offsets = np.arange(0, point_measurement * frame_number,
                                  frame_number)
        time_data = np.zeros_like(frame_offsets)
        path = 'path to large dataset'
        for i, frame_offset in enumerate(frame_offsets):
            print(frame_offset + frame_number)
            t0 = time.time()
            load(os.path.join(path, 'large dataset.emd'),
                 first_frame=frame_offset, last_frame=frame_offset + frame_number)
            t1 = time.time()
            time_data[i] = t1 - t0
        import matplotlib.pyplot as plt
        plt.plot(frame_offsets, time_data)
        plt.xlabel('Frame offset')
        plt.xlabel('Loading time (s)')
