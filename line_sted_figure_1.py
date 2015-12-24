import os
import numpy as np
import matplotlib.pyplot as plt
import np_tif
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit, minimize_scalar

"""
STED microscopy, descanned point vs rescanned line comparison.

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
   pulse fluence of 'F' is expected to excite a fraction (1 - e^(-F)) of
   the ground-state fluorophores it illuminates.
    *('Fluence' means the energy per unit area delivered by one pulse.)
 . Depletion dose is also measured in saturation units. A depletion pulse
   fluence of 'F' is expected to deplete a fraction (1 - e^(-F)) of
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

def main():
    """
    Generate the panels of Figure 1.
    These both take a while to execute.
    """
    if not os.path.exists('Figure_1_output'):
        os.mkdir('Figure_1_output')
    intensity_line_plots()
    sted_dose_plots()

"""
Utility functions
"""
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

def generate_psfs(
    shape, #Desired pixel dimensions of the psfs
    excitation_brightness, #Peak brightness in saturation units
    depletion_brightness, #Peak brightness in saturation units
    blur_sigma,
    psf_type='point',
    save=True,
    verbose=True,
    ):
    """
    A utility function used by psf_report().
    
    Calculate excitation PSFs. Pixel values encode fluence per pulse, in
    saturation units.
    """
    excitation_psf_point = np.zeros(shape) #Used even if pst_type == 'line'
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
    """
    Calculate depletion PSFs. Pixel values encode fluence per pulse, in
    saturation units.
    """
    if psf_type == 'point':
        depletion_psf_point = gaussian_filter(
            excitation_psf_point,
            sigma=blur_sigma)
        depletion_psf_point = (
            (depletion_psf_point / depletion_psf_point.max()) -
            (excitation_psf_point / excitation_psf_point.max()))
        depletion_psf_point *= depletion_brightness  / depletion_psf_point.max()
    elif psf_type == 'line':
        depletion_psf_line = gaussian_filter(
            excitation_psf_line,
            sigma=(0, 0, blur_sigma))
        depletion_psf_line = (
            (depletion_psf_line / depletion_psf_line.max()) -
            (excitation_psf_line / excitation_psf_line.max()))
        depletion_psf_line *= depletion_brightness / depletion_psf_line.max()
    """
    Calculate saturated PSFs. Pixel values encode probability per pulse
    that a ground-state molecule will become excited (excitation) or an
    excited molecule will remain excited (depletion).
    """
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
    """
    Calculate STED PSFs. Pixel values encode probability per pulse that
    a molecule will become excited, but not be depleted.
    """
    if psf_type == 'point':
        sted_psf_point = (saturated_excitation_psf_point *
                          saturated_depletion_psf_point)
    elif psf_type == 'line':
        sted_psf_line = (saturated_excitation_psf_line *
                         saturated_depletion_psf_line)
    """
    Calculate line-rescan STED PSF.
    For descanned point-STED, the system PSF is the STED-shrunk
    excitation PSF. For rescanned line-STED, the system PSF also
    involves the emission PSF.
    """
    if psf_type == 'line':
        emission_sigma = blur_sigma
        line_sted_sigma, _ = get_width(
            sted_psf_line[0, sted_psf_line.shape[1]//2, :])
        rescan_ratio = (emission_sigma / line_sted_sigma)**2 + 1
        if verbose: print(" Rescan ratio: %0.5f"%(rescan_ratio))
        rescan_ratio = int(np.round(rescan_ratio))
        if verbose: print(" Neareset integer:", rescan_ratio)
        point_obj = np.zeros_like(excitation_psf_point)
        point_obj[0, point_obj.shape[1]//2, point_obj.shape[2]//2] = 1
        emission_psf = excitation_psf_point.copy() #No Stokes shift
        rescanned_signal_inst = np.zeros((
            point_obj.shape[0],
            point_obj.shape[1],
            int(rescan_ratio * point_obj.shape[2])))
        rescanned_signal_cumu = rescanned_signal_inst.copy()
        if verbose: print(" Calculating rescan psf...", end='')
        for scan_position in range(point_obj.shape[2]):
            """
            Scan the excitation
            Multiply the object by excitation
            Blur the excited object
            Unroll the blurred excited object
            Roll the blurred excited object to the rescan position
            """
            scanned_excitation = np.roll(
                sted_psf_line, scan_position - point_obj.shape[2]//2, axis=2)
            glow = gaussian_filter(point_obj * scanned_excitation,
                                   sigma=emission_sigma)
            descanned_signal = np.roll(
                glow, point_obj.shape[2]//2 - scan_position, axis=2)
            rescanned_signal_inst.fill(0)
            rescanned_signal_inst[0, :, :point_obj.shape[2]] = descanned_signal
            rescanned_signal_inst = np.roll(
                rescanned_signal_inst,
                scan_position*int(rescan_ratio) - point_obj.shape[2]//2,
                axis=2)
            rescanned_signal_cumu += rescanned_signal_inst
        if verbose: print(" ...done.")
        rescanned_line_sted_psf = np.roll(
            rescanned_signal_cumu, #Roll so center bin is centered on the image
            rescan_ratio // 2,
            axis=2).reshape( #Quick and dirty binning
                1,
                rescanned_signal_cumu.shape[1],
                rescanned_signal_cumu.shape[2] / int(rescan_ratio),
                int(rescan_ratio)
                ).sum(axis=3)
    if save:
        if not os.path.exists('intermediate_psfs'):
            os.mkdir('intermediate_psfs')
        if psf_type == 'point':
            np_tif.array_to_tif(
                excitation_psf_point,
                'intermediate_psfs/excitation_psf_point.tif')
            np_tif.array_to_tif(
                depletion_psf_point,
                'intermediate_psfs/depletion_psf_point.tif')
            np_tif.array_to_tif(
                saturated_excitation_psf_point,
                'intermediate_psfs/excitation_fraction_psf_point.tif')
            np_tif.array_to_tif(
                saturated_depletion_psf_point,
                'intermediate_psfs/depletion_fraction_psf_point.tif')
            np_tif.array_to_tif(
                sted_psf_point,
                'intermediate_psfs/sted_psf_point.tif')
        elif psf_type == 'line':
            np_tif.array_to_tif(
                excitation_psf_line,
                'intermediate_psfs/excitation_psf_line.tif')
            np_tif.array_to_tif(
                depletion_psf_line,
                'intermediate_psfs/depletion_psf_line.tif')
            np_tif.array_to_tif(
                saturated_excitation_psf_line,
                'intermediate_psfs/excitation_fraction_psf_line.tif')
            np_tif.array_to_tif(
                saturated_depletion_psf_line,
                'intermediate_psfs/depletion_fraction_psf_line.tif')
            np_tif.array_to_tif(
                sted_psf_line,
                'intermediate_psfs/sted_psf_line.tif')
            np_tif.array_to_tif(
                emission_psf,
                'intermediate_psfs/emission_psf.tif')
            np_tif.array_to_tif(
                rescanned_signal_cumu,
                'intermediate_psfs/sted_psf_line_rescan_unscaled.tif')
            np_tif.array_to_tif(
                rescanned_line_sted_psf,
                'intermediate_psfs/sted_psf_line_rescan.tif')
    if psf_type == 'point':
        return {
            'excitation': excitation_psf_point,
            'depletion': depletion_psf_point,
            'excitation_fraction': saturated_excitation_psf_point,
            'depletion_fraction': saturated_depletion_psf_point,
            'sted': sted_psf_point}
    elif psf_type == 'line':
        return {
            'excitation': excitation_psf_line,
            'depletion': depletion_psf_line,
            'excitation_fraction': saturated_excitation_psf_line,
            'depletion_fraction': saturated_depletion_psf_line,
            'sted': sted_psf_line,
            'rescan_sted': rescanned_line_sted_psf}

def psf_report(
    illumination_shape, #Point or line
    excitation_brightness, #Peak brightness in saturation units
    depletion_brightness, #Peak brightness in saturation units
    steps_per_excitation_psf_width, #Too small? Bad res. Too big? Excess dose.
    pulses_per_position, #Think of this as "dwell time"
    verbose=True,
    save=False,
    ):
    """
    The primary function of this module. See the start of this file for
    a lengthy description.

    Excitation PSF width (FWHM) defines our length unit, and we've chosen
    how many steps (pixels) we want per excitation PSF. What
    Gaussian 'sigma' does this correspond to?
    """
    blur_sigma = steps_per_excitation_psf_width / (2*np.sqrt(2*np.log(2)))
    num_steps = 1 + 2*int(np.round(5*blur_sigma)) #Should be wide and odd
    psfs = generate_psfs(
        shape=(1, num_steps, num_steps),
        excitation_brightness=excitation_brightness,
        depletion_brightness=depletion_brightness,
        blur_sigma=blur_sigma,
        psf_type=illumination_shape,
        verbose=verbose,
        save=save)
    """
    Calculate STED effective excitation PSF width, in pixels. Also
    calculate the resolution improvement factor compared to confocal.
    """
    central_line_ex = psfs['excitation'][0, num_steps//2, :]
    central_line_st = psfs['sted'][0, num_steps//2, :]
    assert central_line_ex.max() == psfs['excitation'].max()
    assert central_line_st.max() == psfs['sted'].max()
    ex_sigma, _ = get_width(central_line_ex)
    sted_sigma, _ = get_width(central_line_st)
    resolution_improvement_factor =  blur_sigma / sted_sigma
    if verbose:
        print("Illumination shape:", illumination_shape)
        print("Excitation psf width: %0.3f"%(ex_sigma * 2*np.sqrt(2*np.log(2))),
              "pixels FWHM")
        print("STED psf width: %0.3f"%(sted_sigma * 2*np.sqrt(2*np.log(2))),
              "pixels FWHM")
        print("STED improvement in excitation PSF width: %0.3f"%(
            resolution_improvement_factor))
    if illumination_shape == 'line':
        central_line_rescan = psfs['rescan_sted'][0, num_steps//2, :]
        assert central_line_rescan.max() == psfs['rescan_sted'].max()
        rescan_sigma, _ = get_width(central_line_rescan)
        resolution_improvement_factor = blur_sigma / rescan_sigma
        if verbose:
            print("Rescan STED psf width: %0.3f"%(
                rescan_sigma * 2*np.sqrt(2*np.log(2))))
            print("Rescan STED improvement in PSF width: %0.3f"%(
                resolution_improvement_factor))
    """
    Calculate emission levels and dose per pulse from the excitation and
    depletion PSFs. Note that the scan pattern is very important when
    calculating the illumnation dose! Point-sted requires a 2D raster,
    compared to line STED which only needs a 1D scan. This leads to a
    huge difference in dose.
    """
    if illumination_shape == 'point': #2D raster scan
        excitation_dose = psfs['excitation'].sum()
        depletion_dose = psfs['depletion'].sum()
        expected_emissions = psfs['sted'].sum()
    elif illumination_shape == 'line': #1D scan
        excitation_dose = psfs['excitation'][0, num_steps//2, :].sum()
        depletion_dose = psfs['depletion'][0, num_steps//2, :].sum()
        expected_emissions = psfs['sted'][0, num_steps//2, :].sum()
    if verbose:
        print("Excitation dose: %0.3f"%(pulses_per_position * excitation_dose),
              "half-saturations")
        print("Depletion dose: %0.3f"%(pulses_per_position * depletion_dose),
              "half-saturations")
        print("Expected emissions per molecule: %0.4f"%(
            pulses_per_position * expected_emissions))
        print("")
    return {'resolution_improvement': resolution_improvement_factor,
            'excitation_dose': pulses_per_position * excitation_dose,
            'depletion_dose': pulses_per_position * depletion_dose,
            'expected_emission': pulses_per_position * expected_emissions,
            'pulses_per_position': pulses_per_position,
            'psfs': psfs}

def tune_psf(
    illumination_shape, #'point' or 'line'
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

    psf_report() lets us specify illumination intensities and calculates
    resolution, emission, and dose. We want to plot dose vs resolution
    and emission, so we use tune_psf() to find the appropriate
    intensities. Note that you can always get more emission by
    increasing the excitation intensity, but due to saturation, this is
    a terrible approach, so we cap the excitation intensity.

    This function is not fast or efficient, but that suits my needs, it
    just has to generate a few figures.
    """
    steps_per_excitation_psf_width = (steps_per_improved_psf_width *
                                      desired_resolution_improvement)
    args = { #The inputs to psf_report()
        'illumination_shape': illumination_shape,
        'excitation_brightness': max_excitation_brightness,
        'depletion_brightness': 1,
        'steps_per_excitation_psf_width': steps_per_excitation_psf_width,
        'pulses_per_position': 1,
        'verbose': False,
        'save': False,}
    num_iterations = 0
    while True:
        num_iterations += 1
        if num_iterations >= 10:
            print("Max. iterations exceeded; giving up")
            break
        """
        Tune the depletion brightness
        """
        def minimize_me(depletion_brightness):
            args['depletion_brightness'] = abs(depletion_brightness)
            results = psf_report(**args)
            return (results['resolution_improvement'] -
                    desired_resolution_improvement)**2
        args['depletion_brightness'] = abs(minimize_scalar(minimize_me).x)
        if verbose_iterations:
            print("Depletion brightness:", args['depletion_brightness'])
        """
        How many pulses do we need?
        """
        args['excitation_brightness'] = max_excitation_brightness
        args['pulses_per_position'] = 1
        results = psf_report(**args)
        args['pulses_per_position'] = np.ceil((desired_emissions_per_molecule /
                                               results['expected_emission']))
        if verbose_iterations:
            print(args['pulses_per_position'], "pulses.")
        """
        Tune the excitation brightness
        """
        def minimize_me(excitation_brightness):
            args['excitation_brightness'] = abs(excitation_brightness)
            results = psf_report(**args)
            return (results['expected_emission'] -
                    desired_emissions_per_molecule)**2
        args['excitation_brightness'] = abs(minimize_scalar(minimize_me).x)
        if verbose_iterations:
            print("Excitation brightness:", args['excitation_brightness'])
        results = psf_report(**args)
        """
        We already tuned the depletion brightness to get the desired
        resolution, but then we changed the excitation brightness, which
        can slightly affect resolution due to excitation saturation.
        Check to make sure we're still close enough to the desired
        resolution; if not, take it from the top.
        """
        relative_resolution_error = ((results['resolution_improvement'] -
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

"""
Figure generation functions. Complicated and fiddly, but the figures
come out just how I want them. These functions aren't super important to
the conclusions of the project, so I didn't annotate them with comments.
"""
def intensity_line_plots():
    desired_resolution_improvement=3
    desired_emissions_per_molecule=1
    max_excitation_brightness=0.5
    steps_per_improved_psf_width=9
    psfs = {}
    print("Calculating PSFs...")
    point_results = tune_psf(
        illumination_shape='point',
        desired_resolution_improvement=desired_resolution_improvement,
        desired_emissions_per_molecule=desired_emissions_per_molecule,
        max_excitation_brightness=max_excitation_brightness,
        steps_per_improved_psf_width=steps_per_improved_psf_width)
    psfs['point'] = point_results['psfs']
    line_results = tune_psf(
        illumination_shape='line',
        desired_resolution_improvement=desired_resolution_improvement,
        desired_emissions_per_molecule=desired_emissions_per_molecule,
        max_excitation_brightness=max_excitation_brightness,
        steps_per_improved_psf_width=steps_per_improved_psf_width)
    print("Done calculating.")
    psfs['line'] = line_results['psfs']
    np_tif.array_to_tif(
        np.concatenate(
            (psfs['point']['excitation'],
             psfs['point']['depletion']),
            axis=0),
            'point_psfs.tif',
        channels=2, slices=1)
    np_tif.array_to_tif(
        np.concatenate(
            (psfs['line']['excitation'],
             psfs['line']['depletion']),
            axis=0),
            'line_psfs.tif',
        channels=2, slices=1)
    def norm(x):
        return (x - x.min()) / x.max()
    which_line = psfs['point']['excitation'].shape[1] // 2
    fig = plt.figure()
    plt.fill(norm(psfs['point']['excitation'][0, which_line, :]),
             'g', label='Depletion events', alpha=0.3)
    plt.fill(norm(psfs['point']['sted'][0, which_line, :]),
             'g', label='Emission events', alpha=0.9)
    plt.plot(norm(psfs['point']['excitation'][0, which_line, :]),
             'b-', label='Line excitation', linewidth=3)
    plt.plot(norm(psfs['point']['excitation'][0, which_line, :]),
             'bo', label='Point excitation', linewidth=3)
    plt.plot(norm(psfs['line']['depletion'][0, which_line, :]),
             'g-', label='Eclair depletion', linewidth=3)
    plt.plot(norm(psfs['point']['depletion'][0, which_line, :]),
             'go', label='Donut depletion', linewidth=3)
    plt.legend(fontsize=10, loc='lower right')
    plt.savefig('Fig_1_psf_plots.png')
    fig.show()

def sted_dose_plots():
    """
    Plot excitation and depletion dose vs. resolution improvement for
    low and high emissions per molecule.
    """
    print("Calculating...")
    plt.figure()
    plt.suptitle("Line-STED is much gentler than point-STED",
                 fontweight='bold', fontsize=15)
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    emissions_per_molecule = (0.5, 5)
    max_ex = 0.25
    resolution_improvements = (
        1.001, 1.05, 1.1, 1.25, 1.377, 1.38, 1.4, 1.41,
        1.42, 1.45, 1.5, 1.6, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 7)
    for i, e in enumerate(emissions_per_molecule):
        point_excitation_doses = []
        line_excitation_doses = []
        point_depletion_doses = []
        line_depletion_doses = []
        line_resolution_improvements = []
        for R in resolution_improvements:
            point_results = tune_psf(
                illumination_shape='point',
                desired_resolution_improvement=R,
                desired_emissions_per_molecule=e,
                max_excitation_brightness=max_ex,
                steps_per_improved_psf_width=4,
                verbose_results=True,
                verbose_iterations=True)
            point_excitation_doses.append(point_results['excitation_dose'])
            point_depletion_doses.append(point_results['depletion_dose'])
            if R >= 1.37: #Rescanned line-STED can't do much worse
                line_results = tune_psf(
                    illumination_shape='line',
                    desired_resolution_improvement=R,
                    desired_emissions_per_molecule=e,
                    max_excitation_brightness=max_ex,
                    steps_per_improved_psf_width=4,
                    verbose_results=True,
                    verbose_iterations=True)
                print(
                    "For a resolution improvement of %0.2f"%(
                        point_results['resolution_improvement']),
                    "and %0.2f emissions expected per molecule:\n"%(
                        point_results['expected_emission']),
                    "Line STED uses %0.2fx gentler excitation"%(
                        point_results['excitation_dose'] /
                        line_results['excitation_dose']),
                    "and %0.2fx gentler depletion.\n"%(
                        point_results['depletion_dose'] /
                        line_results['depletion_dose']))
                line_excitation_doses.append(
                    line_results['excitation_dose'])
                line_depletion_doses.append(
                    line_results['depletion_dose'] + 1e-16) #Avoid zeros in log
                line_resolution_improvements.append(R)
        ax1.semilogy(resolution_improvements, point_excitation_doses, '--',
                     label='Point-STED, %0.1f emissions/mol'%(e),
                     linewidth=4, markersize=6, color=('b', 'm')[i])
        ax1.semilogy(line_resolution_improvements, line_excitation_doses, '-',
                     label='Line-STED,  %0.1f emissions/mol'%(e),
                     linewidth=4, color=('b', 'm')[i])
        ax1.set_ylim(0.5, 1.5*max(point_excitation_doses))
        ax2.semilogy(resolution_improvements, point_depletion_doses, '--',
                     label='Point-STED, %0.1f emissions/mol'%(e),
                     linewidth=4, markersize=6, color=('b', 'm')[i])
        ax2.semilogy(line_resolution_improvements, line_depletion_doses, '-',
                     label='Line-STED,  %0.1f emissions/mol'%(e),
                     linewidth=4, color=('b', 'm')[i])
        ax2.set_ylim(2.5, 1.5*max(point_depletion_doses))

    ax1.set_xticklabels([])
    ax1.set_ylabel('Excitation fluence\n(half-saturations)',
                   fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid()
    ax1.grid(True, which='minor', axis='y')
    ax2.set_xlabel('Resolution improvement', fontweight='bold')
    ax2.set_ylabel('Depletion fluence\n(half-saturations)',
                   fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid()
    ax2.grid(True, which='minor', axis='y')
    plt.show()

if __name__ == '__main__':
    main()
