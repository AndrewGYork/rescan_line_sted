import os
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import interpolation
from tqdm import tqdm as progressbar
import np_tif
from line_sted_figure_1 import psf_report

"""
Rescanned line STED is much gentler and faster than descanned point
STED, but has very anisotropic resolution. To give isotropic resolution
and comparable image quality to point-STED, line-STED needs to fuse
multiple images taken with different scan directions. This module
compares simulations of several operating points to illustrate that for
the same light dose, fusion of multiple line rescan directions gives
better images than point descanned imaging.
"""

def main():
    output_prefix = os.path.join(os.getcwd(), 'Figure_3_output/')
    """
    Create a deconvolution object for each of the following methods:
     . Bright descanned point-scanning STED
     . Bright rescanned line-scanning STED with eight orientations
     . Gentle descanned point-scanning STED
     . Gentle rescanned line-scanning STED with two orientations
     . Descanned point-scanning non-STED (diffraction limited)
     . Rescanned line-scanning non-STED (diffraction limited, but SIM-like)
    ...using the appropriate psfs.
    """
    psfs = calculate_psfs() #A lot going on in this function; see below.
    deconvolvers = {name: Deconvolver(psf, (output_prefix + name + '_'))
                    for name, psf in psfs.items()}
    """
    Use our test object to create simulated data for each imaging method
    and dump the data to disk:
    """
    test_object = np_tif.tif_to_array('test_object_2.tif').astype(np.float64)
    for decon_object in deconvolvers.values():
        decon_object.create_data_from_object(test_object,
                                             total_brightness=5e10)
        decon_object.record_data()
    """
    Deconvolve each of our objects, saving occasionally:
    """
    input("\nHit enter to begin deconvolution.")
    print('Deconvolving...')
    num_iterations = 2**14 + 1
    save_these_iterations = [
        2**_ for _ in range(int(np.log2(num_iterations)))]
    for i in progressbar(range(num_iterations)):
        if i in save_these_iterations:
            print("\rSaving iteration", i, ' '*7)
        for decon_object in deconvolvers.values():
            decon_object.iterate()
            if i in save_these_iterations:
                decon_object.record_iteration()
    print('Done deconvolving.')

def calculate_psfs():
    """
    Tune a family of comparable psfs, one each for:
     . Bright descanned point-scanning STED
     . Bright rescanned line-scanning STED with eight orientations
     . Gentle descanned point-scanning STED
     . Gentle rescanned line-scanning STED with two orientations
     . Descanned point-scanning non-STED (diffraction limited)
     . Rescanned line-scanning non-STED (diffraction limited, but SIM-like)
    """
    print("Calculating bright point-STED psf...")
    bright_point = psf_report(illumination_shape='point',
                              excitation_brightness=0.0613,
                              depletion_brightness=6.897154,
                              steps_per_excitation_psf_width=12,
                              pulses_per_position=1,
                              verbose=True)
    bright_point_sted_psf = [bright_point['psfs']['sted'] *
                             bright_point['pulses_per_position']]
    print("Calculating line-STED psf with eight orientations...")
    line_8 = psf_report(illumination_shape='line',
                        excitation_brightness=0.09785,
                        depletion_brightness=26.41405,
                        steps_per_excitation_psf_width=12,
                        pulses_per_position=8,
                        verbose=True)
    line_sted_8_psfs = [0.125 * rotate(line_8['psfs']['rescan_sted'], angle)
                        for angle in (0, 90, -45, 45, 22.5, -22.5, 112.5, -112.5)]
    print("Calculating gentle point-STED psf...")
    gentle_point = psf_report(illumination_shape='point',
                              excitation_brightness=0.03065,
                              depletion_brightness=0.45981,
                              steps_per_excitation_psf_width=12,
                              pulses_per_position=1,
                              verbose=True)
    gentle_point_sted_psf = [gentle_point['psfs']['sted'] *
                             gentle_point['pulses_per_position']]
    print("Calculating line-STED psf with two orientations...")
    line_2 = psf_report(illumination_shape='line',
                        excitation_brightness=0.1957,
                        depletion_brightness=7.0437,
                        steps_per_excitation_psf_width=12,
                        pulses_per_position=2,
                        verbose=True)
    line_sted_2_psfs = [0.5 * rotate(line_2['psfs']['rescan_sted'], angle)
                        for angle in (0, 90)]
    print("Calculating diffraction-limited psf...")
    non_sted = psf_report(illumination_shape='point',
                          excitation_brightness=0.01839,
                          depletion_brightness=0,
                          steps_per_excitation_psf_width=12,
                          pulses_per_position=1,
                          verbose=True)
    diffraction_limited_psf = [non_sted['psfs']['sted'] *
                               non_sted['pulses_per_position']]
    print("Calculating line psf with two orientations...")
    non_sted_line = psf_report(illumination_shape='line',
                                 excitation_brightness=0.1174,
                                 depletion_brightness=0,
                                 steps_per_excitation_psf_width=12,
                                 pulses_per_position=2,
                                 verbose=True)
    non_sted_line_psfs = [
        0.5 * rotate(non_sted_line['psfs']['rescan_sted'], angle)
        for angle in (0, 90)]
    print("Done calculating.\n")
    print("Light dose (saturation units):")
    print(".  bright point-STED: %0.2f (excitation), %0.2f (depletion)"%(
        bright_point['excitation_dose'],
        bright_point['depletion_dose']))
    print(". bright 8-line-STED: %0.2f (excitation), %0.2f (depletion)"%(
        line_8['excitation_dose'],
        line_8['depletion_dose']))
    print(".  gentle point-STED:  %0.2f (excitation),  %0.2f (depletion)"%(
        gentle_point['excitation_dose'],
        gentle_point['depletion_dose']))
    print(". gentle 2-line-STED:  %0.2f (excitation),  %0.2f (depletion)"%(
        line_2['excitation_dose'],
        line_2['depletion_dose']))
    print(".            non-STED: %0.2f (excitation),    %0.2f (depletion)"%(
        non_sted['excitation_dose'],
        non_sted['depletion_dose']))
    print(".      non-STED line:  %0.2f (excitation),    %0.2f (depletion)"%(
        non_sted_line['excitation_dose'],
        non_sted_line['depletion_dose']))
    return {'bright_point_sted': bright_point_sted_psf,
            'line_8angles_sted': line_sted_8_psfs,
            'gentle_point_sted': gentle_point_sted_psf,
            'line_2angles_sted': line_sted_2_psfs,
            'diffraction_limit': diffraction_limited_psf,
            'line_no_depletion': non_sted_line_psfs}

def rotate(x, degrees):
    if degrees == 0:
        return x
    elif degrees == 90:
        return np.rot90(np.squeeze(x)).reshape(x.shape)
    else:
        return np.clip(
            interpolation.rotate(x, angle=degrees, axes=(1, 2), reshape=False),
            0, 1.1 * x.max())

class Deconvolver:
    def __init__(self, psfs, output_prefix=None, verbose=True):
        """
        'psfs' is a list of numpy arrays, one for each PSF.
        'output_prefix' specifies where the deconvolver object will save files.
        """
        self.psfs = list(psfs)
        if output_prefix is None:
            output_prefix = os.getcwd()
        if not os.path.exists(os.path.dirname(output_prefix)):
            os.mkdir(os.path.dirname(output_prefix))
        self.output_prefix = output_prefix
        self.verbose = verbose
        self.num_iterations = 0
        self.saved_iterations = []
        self.estimate_history = []
        return None

    def create_data_from_object(
        self,
        obj,
        total_brightness=None,
        random_seed=None
        ):
        assert len(obj.shape) == 3
        assert obj.dtype == np.float64
        self.true_object = obj.copy()
        if total_brightness is not None:
            self.true_object *= total_brightness / self.true_object.sum()
        self.noiseless_measurement = self.H(self.true_object)
        if random_seed is not None:
            np.random.seed(random_seed)
        self.noisy_measurement = [np.random.poisson(m) + 1e-9 #No ints, no zeros
                                  for m in self.noiseless_measurement]
        return None

    def load_data_from_tif(self, filename):
        self.noisy_measurement = np_tif.tif_to_array(filename) + 1e-9
        assert self.noisy_measurement.shape == 3
        assert self.noisy_measurement.min() >= 0
        return None

    def iterate(self):
        if self.num_iterations == 0:
            self.estimate = np.ones_like(self.H_t(self.noisy_measurement))
        self.num_iterations += 1
        measurement, estimate, H, H_t = (
            self.noisy_measurement, self.estimate, self.H, self.H_t)
        expected_measurement = H(estimate)
        ratio = [measurement[i] / expected_measurement[i]
                 for i in range(len(measurement))]
        correction_factor = self.H_t(ratio)
        self.estimate *= correction_factor
        return None

    def record_iteration(self, save_tifs=True):
        self.saved_iterations.append(self.num_iterations)
        self.estimate_history.append(self.estimate.copy())
        if save_tifs:
            eh = np.squeeze(np.concatenate(self.estimate_history, axis=0))
            np_tif.array_to_tif(eh, self.output_prefix + 'estimate_history.tif')
            def f(x):
                if len(x.shape) == 2:
                    x = x.reshape(1, x.shape[0], x.shape[1])
                return np.log(1 + np.abs(np.fft.fftshift(
                    np.fft.fftn(x, axes=(1, 2)),
                    axes=(1, 2))))
            np_tif.array_to_tif(
                f(eh - self.true_object),
                self.output_prefix + 'estimate_FT_error_history.tif')
        return None

    def record_data(self):
        if hasattr(self, 'psfs'):
            psfs = np.squeeze(np.concatenate(self.psfs, axis=0))
            np_tif.array_to_tif(psfs, self.output_prefix + 'psfs.tif')
        if hasattr(self, 'true_object'):
            np_tif.array_to_tif(
                self.true_object, self.output_prefix + 'object.tif')
        if hasattr(self, 'noiseless_measurement'):
            nm = np.squeeze(np.concatenate(self.noiseless_measurement, axis=0))
            np_tif.array_to_tif(
                nm, self.output_prefix + 'noiseless_measurement.tif')
        if hasattr(self, 'noisy_measurement'):
            nm = np.squeeze(np.concatenate(self.noisy_measurement, axis=0))
            np_tif.array_to_tif(
                nm, self.output_prefix + 'noisy_measurement.tif')
        return None

    def H(self, x):
        """
        Expected noiseless measurement operator H. If 'x' is our true
        object and there's no noise, H(x) is the measurement we expect.
        """
        result = []
        for p in self.psfs:
            blurred_glow = fftconvolve(x, p, mode='same')
            blurred_glow[blurred_glow < 0] = 0 #fft can give tiny negative vals
            result.append(blurred_glow)
        return result

    def H_t(self, y, normalize=True):
        """
        The transpose of the H operator. By default we normalize
        H_t(ones) == ones, so that RL deconvolution can converge.
        """
        result = np.zeros(y[0].shape)
        for i in range(len(y)):
            blurred_ratio = fftconvolve(y[i], self.psfs[i], mode='same')
            blurred_ratio[blurred_ratio < 0] = 0 #fft can give tiny negative vals
            result += blurred_ratio
        if normalize:
            if not hasattr(self, 'H_t_normalization'):
                self.H_t_normalization = self.H_t([np.ones(_.shape) for _ in y],
                                                  normalize=False)
            result /= self.H_t_normalization
        return result

if __name__ == '__main__':
    main()
