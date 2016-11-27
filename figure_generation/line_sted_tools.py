#!/usr/bin/env python3
# Dependencies from the Python 3 standard library:
import os
import time
# Dependencies from the Scipy stack https://www.scipy.org/stackspec.html :
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, interpolation
from scipy.optimize import curve_fit, minimize_scalar
from scipy.signal import fftconvolve
# Dependencies from https://github.com/AndrewGYork/rescan_line_sted :
import np_tif
"""
Tools for comparison of descanned point STED microscopy vs rescanned
line STED microscopy.

Don't run this script directly, it won't do anything. This module is a
dependency of line_sted_figure_1.py and line_sted_figure_2.py.

Assumptions:
 . 2D simulation
 . Excitation and depletion are pulsed
 . Pulse durations are short compared to fluorescent lifetime (meaning,
   pulse fluence is all that matters)
 . Pulse repetition rate is long compared to fluorescent (and triplet)
   lifetime (meaning, I'm not going to worry about triplet-state
   excitatation, effects relevant to T-REX are orthogonal to the method
   we're describing)
 . Emission PSF is Gaussian (Exact PSF shape is not an interesting detail)
 . Excitation illumination shape is Gaussian, same width as emission PSF
   (this neglects the Stokes shift)
 . Depletion illumination shape is difference-of-Gaussians, inner
   Gaussian is the same width as excitation shape (this is at least in
   the right ballpark, and doesn't exceed the bandwidth of the
   excitation/emission PSFs)
 . 100% of excited molecules that are allowed to emit produce a
   photoelectron on the detector. Not true, but gives results
   proportional to the truth.
Units:
 . Distance is measured in emission PSF widths. Emission PSF full-width
   at half-maximum is 1 unit wide, by definition.
 . Excitation fluence* is measured in saturation units. An excitation
   pulse fluence of 'F' is expected to excite a fraction (1 - 2^(-F)) of
   the ground-state fluorophores it illuminates.
    *('Fluence' means the energy per unit area delivered by one pulse.)
 . Depletion dose is also measured in saturation units. A depletion pulse
   fluence of 'F' is expected to deplete a fraction (1 - 2^(-F)) of
   excited molecules it touches.
 . Time is measured in number of excitation/depletion pulses, since
   signal emission is probably* the relevant speed limit, and pulses are
   spaced by at least the fluorescent lifetime. Neither mechanical
   scanning nor available illumination intensity are likely to limit
   speed, nor is available laser intensity.
    *(Pixels per second is potentially a practical speed limit, and
    worth considering, but probably favors rescan line STED over
    descanned point STED.)
Inputs to psf_report():
 . Descanned point or rescanned line simulation
 . Excitation pulse energy (fluence integrated over excitation area)
 . Depletion pulse energy (fluence integrated over depletion area)
 . Scan step size (Should probably be determined by depeletion fluence
   and Nyquist)
 . Number of excitation/depletion pulses delivered per scan position.
   Increase this to increase emitted signal, but at the cost of
   increasing both the excitaiton dose and the depletion dose.
   Alternatively, you could increase the excitation fluence to incrase
   emitted signal, but of course this saturates at high fluence.
Outputs:
 . PSF. The PSF shape encodes the resolution, and the PSF magnitude
   encodes the brightness. Units of brightness are expected number of
   emission events per molecule.
 . Excitation dose summed over the scan, in saturation units.
 . Depletion dose summed over the scan, in saturation units.
"""
def psf_report(
    psf_type, #Point or line
    excitation_brightness, #Peak brightness in saturation units
    depletion_brightness, #Peak brightness in saturation units
    steps_per_excitation_psf_width, #Too small? Bad res. Too big? Excess dose.
    pulses_per_position, #Think of this as "dwell time"
    verbose=True,
    output_dir=None,
    ):
    """
    The primary function of this module. See the start of this file for
    a lengthy list of simulation assumptions.
    """
    # Excitation PSF width (FWHM) defines our length unit, and we've
    # chosen how many steps (pixels) we want per excitation PSF. What
    # Gaussian 'sigma' does this correspond to?
    blur_sigma = steps_per_excitation_psf_width / (2*np.sqrt(2*np.log(2)))
    num_steps = 1 + 2*int(np.round(5*blur_sigma)) #Should be wide and odd
    psfs = generate_psfs(
        shape=(1, num_steps, num_steps),
        excitation_brightness=excitation_brightness,
        depletion_brightness=depletion_brightness,
        blur_sigma=blur_sigma,
        psf_type=psf_type,
        verbose=verbose,
        output_dir=output_dir)
    # Calculate STED effective excitation PSF width, in pixels. Also
    # calculate the resolution improvement factor compared to confocal.
    central_line_ex = psfs['excitation'][0, num_steps//2, :]
    central_line_st = psfs['sted'][0, num_steps//2, :]
    assert central_line_ex.max() == psfs['excitation'].max()
    assert central_line_st.max() == psfs['sted'].max()
    ex_sigma, _ = get_width(central_line_ex)
    sted_sigma, _ = get_width(central_line_st)
    resolution_improvement_factor_descanned =  blur_sigma / sted_sigma
    if verbose:
        print("PSF type:", psf_type)
        print("Excitation psf width: %0.3f"%(ex_sigma * 2*np.sqrt(2*np.log(2))),
              "pixels FWHM")
        print("STED psf width: %0.3f"%(sted_sigma * 2*np.sqrt(2*np.log(2))),
              "pixels FWHM")
        print("STED improvement in excitation PSF width: %0.3f"%(
            resolution_improvement_factor_descanned))
    if psf_type == 'line':
        central_line_rescan = psfs['rescan_sted'][0, num_steps//2, :]
        assert central_line_rescan.max() == psfs['rescan_sted'].max()
        rescan_sigma, _ = get_width(central_line_rescan)
        resolution_improvement_factor_rescanned = blur_sigma / rescan_sigma
        if verbose:
            print("Rescan STED psf width: %0.3f"%(
                rescan_sigma * 2*np.sqrt(2*np.log(2))))
            print("Rescan STED improvement in PSF width: %0.3f"%(
                resolution_improvement_factor_rescanned))
    # Calculate emission levels and dose per pulse from the excitation
    # and depletion PSFs. Note that the scan pattern is very important
    # when calculating the illumnation dose! Point-sted requires a 2D
    # raster, compared to line STED which only needs a 1D scan. This
    # leads to a huge difference in dose, and a small difference in
    # emission levels.
    if psf_type == 'point': #2D scan, sum an area
        excitation_dose = pulses_per_position * psfs['excitation'].sum()
        depletion_dose = pulses_per_position * psfs['depletion'].sum()
        expected_emissions = pulses_per_position * psfs['sted'].sum()
    elif psf_type == 'line': #1D scan, sum a line
        excitation_dose = (pulses_per_position *
                           psfs['excitation'][0, num_steps//2, :].sum())
        depletion_dose = (pulses_per_position *
                          psfs['depletion'][0, num_steps//2, :].sum())
        expected_emissions = (pulses_per_position *
                              psfs['sted'][0, num_steps//2, :].sum())
    if verbose:
        print("Excitation dose: %0.3f"%(excitation_dose), "half-saturations")
        print("Depletion dose: %0.3f"%(depletion_dose), "half-saturations")
        print("Expected emissions per molecule: %0.4f\n"%(expected_emissions))
    if psf_type == 'point':
        return {'resolution_improvement_descanned':
                resolution_improvement_factor_descanned,
                'excitation_dose': excitation_dose,
                'depletion_dose': depletion_dose,
                'expected_emission': expected_emissions,
                'pulses_per_position': pulses_per_position,
                'psfs': psfs}
    elif psf_type == 'line':
        return {'resolution_improvement_rescanned':
                resolution_improvement_factor_rescanned,
                'resolution_improvement_descanned':
                resolution_improvement_factor_descanned,
                'excitation_dose': excitation_dose,
                'depletion_dose': depletion_dose,
                'expected_emission': expected_emissions,
                'pulses_per_position': pulses_per_position,
                'psfs': psfs}

def generate_psfs(
    shape, #Desired pixel dimensions of the psfs
    excitation_brightness, #Peak brightness in saturation units
    depletion_brightness, #Peak brightness in saturation units
    blur_sigma,
    psf_type='point',
    output_dir=None,
    verbose=True,
    ):
    """
    A utility function used by psf_report().
    """
    # Calculate gaussian point and line excitation patterns. Pixel
    # values encode fluence per pulse, in saturation units.
    if psf_type == 'point':
        excitation_psf_point = np.zeros(shape)
        excitation_psf_point[0, shape[1]//2, shape[2]//2] = 1
        excitation_psf_point = gaussian_filter(excitation_psf_point,
                                               sigma=blur_sigma)
        excitation_psf_point *= excitation_brightness / excitation_psf_point.max()
    if psf_type == 'line':
        excitation_psf_line = np.zeros(shape)
        excitation_psf_line[0, :, shape[2]//2] = 1
        excitation_psf_line = gaussian_filter(excitation_psf_line,
                                              sigma=(0, 0, blur_sigma))
        excitation_psf_line *= excitation_brightness / excitation_psf_line.max()
    # Calculate difference-of-gaussian point and line depletion
    # patterns. Pixel values encode fluence per pulse, in saturation
    # units.
    if psf_type == 'point':
        depletion_psf_inner = np.zeros(shape)
        depletion_psf_inner[0, shape[1]//2, shape[2]//2] = 1
        depletion_psf_inner = gaussian_filter(depletion_psf_inner,
                                              sigma=blur_sigma)
        depletion_psf_outer = gaussian_filter(depletion_psf_inner,
                                              sigma=blur_sigma)
        depletion_psf_point = (
            (depletion_psf_outer / depletion_psf_outer.max()) -
            (depletion_psf_inner / depletion_psf_inner.max()))
        depletion_psf_point *= depletion_brightness  / depletion_psf_point.max()
    elif psf_type == 'line':
        depletion_psf_inner = np.zeros(shape)
        depletion_psf_inner[0, :, shape[2]//2] = 1
        depletion_psf_inner = gaussian_filter(depletion_psf_inner,
                                              sigma=(0, 0, blur_sigma))
        depletion_psf_outer = gaussian_filter(depletion_psf_inner,
                                              sigma=(0, 0, blur_sigma))
        depletion_psf_line = (
            (depletion_psf_outer / depletion_psf_outer.max()) -
            (depletion_psf_inner / depletion_psf_inner.max()))
        depletion_psf_line *= depletion_brightness / depletion_psf_line.max()
    # Calculate "saturated" excitation/depletion patterns. Pixel values
    # encode probability per pulse that a ground-state molecule will
    # become excited (excitation) or an excited molecule will remain
    # excited (depletion).
    half_on_dose = 1
    half_off_dose = 1
    if psf_type == 'point':
        saturated_excitation_psf_point = 1 - 2**(-excitation_psf_point /
                                                 half_on_dose)
        saturated_depletion_psf_point = 2**(-depletion_psf_point /
                                            half_off_dose)
    elif psf_type == 'line':
        saturated_excitation_psf_line = 1 - 2**(-excitation_psf_line /
                                                half_on_dose)
        saturated_depletion_psf_line = 2**(-depletion_psf_line /
                                           half_off_dose)
    # Calculate post-depletion excitation patterns. Pixel values encode
    # probability per pulse that a molecule will become excited, but not
    # be depleted.
    if psf_type == 'point':
        sted_psf_point = (saturated_excitation_psf_point *
                          saturated_depletion_psf_point)
    elif psf_type == 'line':
        sted_psf_line = (saturated_excitation_psf_line *
                         saturated_depletion_psf_line)
    # Calculate the "system" PSF, which can depend on both excitation
    # and emission. For descanned point-STED, the system PSF is the
    # (STED-shrunk) excitation PSF. For rescanned line-STED, the system
    # PSF also involves the emission PSF.
    if psf_type == 'point':
        descanned_point_sted_psf = sted_psf_point # Simple rename
    elif psf_type == 'line':
        emission_sigma = blur_sigma # Assume emission PSF same as excitation PSF
        line_sted_sigma, _ = get_width(
            sted_psf_line[0, sted_psf_line.shape[1]//2, :])
        line_rescan_ratio = (emission_sigma / line_sted_sigma)**2 + 1
        if verbose: print(" Ideal line rescan ratio: %0.5f"%(line_rescan_ratio))
        line_rescan_ratio = int(np.round(line_rescan_ratio))
        if verbose: print(" Neareset integer:", line_rescan_ratio)
        point_obj = np.zeros(shape)
        point_obj[0, point_obj.shape[1]//2, point_obj.shape[2]//2] = 1
        emission_psf = gaussian_filter(point_obj, sigma=emission_sigma)
        rescanned_signal_inst = np.zeros((
            point_obj.shape[0],
            point_obj.shape[1],
            int(line_rescan_ratio * point_obj.shape[2])))
        rescanned_signal_cumu = rescanned_signal_inst.copy()
        descanned_signal_cumu = np.zeros(shape)
        if verbose: print(" Calculating rescan psf...", end='')
        # I could use an analytical shortcut to calculate the rescan PSF
        # (see http://dx.doi.org/10.1364/BOE.4.002644 for details), but
        # for the sake of clarity, I explicitly simulate the rescan
        # imaging process:
        for scan_position in range(point_obj.shape[2]):
            # . Scan the excitation
            scanned_excitation = np.roll(
                sted_psf_line, scan_position - point_obj.shape[2]//2, axis=2)
            # . Multiply the object by excitation to calculate the "glow":
            glow = point_obj * scanned_excitation
            # . Blur the glow by the emission PSF to calculate the image
            #   on the detector:
            blurred_glow = gaussian_filter(glow,  sigma=emission_sigma)
            # . Calculate the contribution to the descanned image (the
            #   kind measured by Curdt or Schubert
            #   http://www.ub.uni-heidelberg.de/archiv/14362
            #   http://www.ub.uni-heidelberg.de/archiv/15986
            descanned_signal = np.roll(
                blurred_glow, point_obj.shape[2]//2 - scan_position, axis=2)
            descanned_signal_cumu[:, :, scan_position
                                  ] += descanned_signal.sum(axis=2)
            # . Roll the descanned image to the rescan position, to
            #   produce the "instantaneous" rescanned image:
            rescanned_signal_inst.fill(0)
            rescanned_signal_inst[0, :, :point_obj.shape[2]] = descanned_signal
            rescanned_signal_inst = np.roll(
                rescanned_signal_inst,
                scan_position * line_rescan_ratio - point_obj.shape[2]//2,
                axis=2)
            # . Add the "instantaneous" image to the "cumulative" image.
            rescanned_signal_cumu += rescanned_signal_inst
        if verbose: print(" ...done.")
        # . Bin the rescanned psf back to the same dimensions as the object:
        rescanned_line_sted_psf = np.roll(
            rescanned_signal_cumu, #Roll so center bin is centered on the image
            int(line_rescan_ratio // 2),
            axis=2).reshape( #Quick and dirty binning
                1,
                rescanned_signal_cumu.shape[1],
                int(rescanned_signal_cumu.shape[2] / line_rescan_ratio),
                int(line_rescan_ratio)
                ).sum(axis=3)
        descanned_line_sted_psf = descanned_signal_cumu # Simple rename
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if psf_type == 'point':
            for array, filename in (
                (excitation_psf_point,
                 'excitation_psf_point.tif'),
                (depletion_psf_point,
                 'depletion_psf_point.tif'),
                (saturated_excitation_psf_point,
                 'excitation_fraction_psf_point.tif'),
                (saturated_depletion_psf_point,
                 'depletion_fraction_psf_point.tif'),
                (sted_psf_point,
                 'sted_psf_point.tif')):
                np_tif.array_to_tif(array, os.path.join(output_dir, filename))
        elif psf_type == 'line':
            for array, filename in (
                (excitation_psf_line,
                 'excitation_psf_line.tif'),
                (depletion_psf_line,
                 'depletion_psf_line.tif'),
                (saturated_excitation_psf_line,
                 'excitation_fraction_psf_line.tif'),
                (saturated_depletion_psf_line,
                 'depletion_fraction_psf_line.tif'),
                (sted_psf_line,
                 'sted_psf_line.tif'),
                (emission_psf, 'emission_psf.tif'),
                (rescanned_signal_cumu,
                 'sted_psf_line_rescan_unscaled.tif'),
                (rescanned_line_sted_psf,
                 'sted_psf_line_rescan.tif'),
                (descanned_line_sted_psf,
                 'sted_psf_line_descan.tif')):
                np_tif.array_to_tif(array, os.path.join(output_dir, filename))
    if psf_type == 'point':
        return {
            'excitation': excitation_psf_point,
            'depletion': depletion_psf_point,
            'excitation_fraction': saturated_excitation_psf_point,
            'depletion_fraction': saturated_depletion_psf_point,
            'sted': sted_psf_point,
            'descan_sted': descanned_point_sted_psf}
    elif psf_type == 'line':
        return {
            'excitation': excitation_psf_line,
            'depletion': depletion_psf_line,
            'excitation_fraction': saturated_excitation_psf_line,
            'depletion_fraction': saturated_depletion_psf_line,
            'sted': sted_psf_line,
            'descan_sted': descanned_line_sted_psf,
            'rescan_sted': rescanned_line_sted_psf}

def tune_psf(
    psf_type, #'point' or 'line'
    scan_type, # 'descanned' or 'rescanned'
    desired_resolution_improvement,
    desired_emissions_per_molecule,
    max_excitation_brightness=0.5, #Saturation units
    steps_per_improved_psf_width=3,
    relative_error=1e-6,
    verbose_results=False,
    verbose_iterations=False,
    ):
    """
    Find the combination of excitation brightness, depletion brightness,
    and number of pulses to give the desired resolution improvement and
    emissions per molecule, without exceeding a limit on excitation
    brightness.

    psf_report() lets us specify illumination intensities and scan step
    sizes, and calculates resolution, emission, and dose. That's
    conceptually great; these are the parameters we'd hand-tune when
    using a STED microscope. Unfortunately, this also lets us make all
    kinds of bad choices. For example, unless the step size matches the
    resolution improvement, there's tons of excess dose for no gain. For
    another example, you can always get more emission by increasing the
    excitation intensity, but due to saturation, this is a terrible
    approach

    tune_psf() lets us specify our goals (resolution, emitted signal),
    and figures out the inputs needed to get there.

    This function is not fast or efficient, but that suits my needs, it
    just has to generate a few figures.
    """
    assert (psf_type, scan_type) in (('point', 'descanned'),
                                     ('line', 'descanned'),
                                     ('line', 'rescanned'))
    assert float(desired_resolution_improvement) == desired_resolution_improvement
    assert float(desired_emissions_per_molecule) == desired_emissions_per_molecule
    assert float(max_excitation_brightness) == max_excitation_brightness
    assert float(steps_per_improved_psf_width) == steps_per_improved_psf_width
    assert float(relative_error) == relative_error
    steps_per_excitation_psf_width = (steps_per_improved_psf_width *
                                      desired_resolution_improvement)
    args = { #The inputs to psf_report()
        'psf_type': psf_type,
        'excitation_brightness': max_excitation_brightness,
        'depletion_brightness': 1,
        'steps_per_excitation_psf_width': steps_per_excitation_psf_width,
        'pulses_per_position': 1,
        'verbose': False,
        'output_dir': None}
    num_iterations = 0
    while True:
        num_iterations += 1
        if num_iterations >= 10:
            print("Max. iterations exceeded; giving up")
            break
        # Tune the depletion brightness to get the desired resolution
        # improvement:
        def minimize_me(depletion_brightness):
            args['depletion_brightness'] = abs(depletion_brightness)
            results = psf_report(**args)
            return (results['resolution_improvement_' + scan_type] -
                    desired_resolution_improvement)**2
        args['depletion_brightness'] = abs(minimize_scalar(minimize_me).x)
        if verbose_iterations:
            print("Depletion brightness:", args['depletion_brightness'])
        # How many excitation/depletion pulse pairs do we need to
        # achieve the desired emission?
        args['excitation_brightness'] = max_excitation_brightness
        args['pulses_per_position'] = 1
        results = psf_report(**args) # Calculate emission from one pulse
        args['pulses_per_position'] = np.ceil((desired_emissions_per_molecule /
                                               results['expected_emission']))
        if verbose_iterations:
            print(args['pulses_per_position'], "pulses.")
        # Tune the excitation brightness to get the desired emissions
        # per molecule:
        def minimize_me(excitation_brightness):
            args['excitation_brightness'] = abs(excitation_brightness)
            results = psf_report(**args)
            return (results['expected_emission'] -
                    desired_emissions_per_molecule)**2
        args['excitation_brightness'] = abs(minimize_scalar(minimize_me).x)
        if verbose_iterations:
            print("Excitation brightness:", args['excitation_brightness'])
        results = psf_report(**args)
        ## We already tuned the depletion brightness to get the desired
        ## resolution, but then we changed the excitation brightness,
        ## which can slightly affect resolution due to excitation
        ## saturation. Check to make sure we're still close enough to
        ## the desired resolution; if not, take it from the top.
        relative_resolution_error = (
            (results['resolution_improvement_' + scan_type] -
             desired_resolution_improvement) /
            desired_resolution_improvement)
        if (relative_resolution_error < relative_error):
            break
    if verbose_results:
        print("PSF tuning complete, after", num_iterations, "iterations.")
        print(" Inputs:")
        for k in sorted(args.keys()):
            print('  ', k, ': ', args[k], sep='')
        print(" Outputs:")
        for k in sorted(results.keys()):
            if k == 'psfs':
                print('  ', k, ': ', sorted(results[k].keys()), sep='')
            else:
                print('  ', k, ': ', results[k], sep='')
        print()        
    results.update(args) #Combine the two dictionaries
    return results

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

def logarithmic_progress(iterable, verbose=True):
    """A funny type of progress bar I use a lot.

    This does two things:
    * Return a true/false flag if the iteration is a power of two, or
    the last iteration. Usually I want to save results at this point, or
    print an update to screen.
    * Print a simple progress bar that works in most terminals, if
      'verbose', but only after the iteration has gone for a few seconds.

    For an example of why I might want such a thing, compare the
    following three loops:
    for i in range(1000):
        time.sleep(0.01)
        print(i, "(too much spam!)")
    for i, save_i in logarithmic_progress(range(1000), verbose=False):
        time.sleep(0.01)
        if save_i: print(i, "(Less spam, but too much suspense!)")
    for i, save_i in logarithmic_progress(range(1000)):
        time.sleep(0.01)
        if save_i: print(i, "(My preferred compromise)")
    """
    if len(iterable) == 0: return iterable
    i, save = 0, []
    while 2**i + 1 < len(iterable):
        save.append(2**i) # A list of the powers of 2 less than len(iterable)
        i += 1
    save.append(len(iterable) - 1) # That ends in iterable[-1]
    bar = "Progress:\n|0%"+" "*13+"|"+" "*15+"|50%"+" "*12+"|"+" "*15 + "|100%"
    i = -1
    bar_printed = False
    stars_printed = 0
    start_time = time.perf_counter()
    for x in iterable:
        i += 1
        yield (x, i in save)
        if verbose: # Maybe print or update a progress bar
            elapsed = time.perf_counter() - start_time
            if elapsed > 1.5:
                if i in save:
                    rate = i / elapsed
                    remaining = (len(iterable) - i) / rate
                    print("Iteration ", i, "/", len(iterable) - 1,
                          " %0.1fs elapsed, "%(elapsed),
                          "~%0.1fs remaining, "%(remaining),
                          "%0.1f iter/s\n"%(rate),
                          sep='', end='')
                    bar_printed = False
                    stars_printed = 0
                if not bar_printed:
                    print(bar)
                    bar_printed = True
                while stars_printed/65 < i/len(iterable):
                    print("*", end='')
                    stars_printed += 1
                if (i+1) in save: print()

def get_width(x):
    """
    Measure the "width" of x by fitting a Gaussian. Quick-and-dirty.
    """
    def gauss(x, *p):
        A, mu, sigma = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))
    coords = range(len(x))
    coeff, _ = curve_fit(
        gauss,
        coords,
        x,
        p0=[1., len(x) / 2., 1.]) # Initial guess for A, mu and sigma
    fit = gauss(coords, *coeff)
    width = coeff[2]
    return width, fit

