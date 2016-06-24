import os
import numpy as np
from scipy.ndimage import interpolation
import np_tif
import line_sted_tools as st
"""
Line STED is much gentler than point STED, but has very anisotropic
resolution. To give isotropic resolution and comparable image quality to
point-STED, line-STED needs to fuse multiple images taken with different
scan directions. This module compares simulations of several operating
points to illustrate that for the same image quality, fusion of multiple
line scan directions gives lower light dose.
"""
def main():
    output_prefix = os.path.join(os.getcwd(), 'Figure_2_output/')
    # Create a deconvolution object for each of the psfs created below
    psfs = calculate_psfs() #A lot going on in this function; see below.
    deconvolvers = {name: st.Deconvolver(psf, (output_prefix + name + '_'))
                    for name, psf in psfs.items()}
    # Use our test object to create simulated data for each imaging
    # method and dump the data to disk:
    test_object = np_tif.tif_to_array('test_object_2.tif').astype(np.float64)
    for decon_object in deconvolvers.values():
        decon_object.create_data_from_object(test_object, total_brightness=5e10)
        decon_object.record_data()
    # Deconvolve each of our objects, saving occasionally:
    input("\nHit enter to begin deconvolution.")
    print('Deconvolving...')
    num_iterations = 2**10 + 1
    for i, save in st.logarithmic_progress(range(num_iterations)):
        for decon_object in deconvolvers.values():
            decon_object.iterate()
            if save:
                decon_object.record_iteration()
    print('\nDone deconvolving.')

def calculate_psfs():
    """
    Tune a family of comparable psfs, one each for:
     . Bright point-STED
     . Bright line-STED with eight orientations
     . Medium point-STED
     . Medium line-STED with four orientations
     . Gentle point-STED
     . Gentle line-STED with two orientations
     . Point non-STED (diffraction limited)
     . Line non-STED (diffraction limited)
    """
    psfs = {}
##    psfs['gentle'] = psf_comparison_pair(
##        point_resolution_improvement=2,
##        line_resolution_improvement=4.04057,
##        point_emissions_per_molecule=4,
##        line_emissions_per_molecule=3.007,
##        line_scan_type='descanned',
##        line_num_orientations=4)
    psfs['gentle'] = psf_comparison_pair(
        point_resolution_improvement=2.5,
        line_resolution_improvement=5.13325,
        point_emissions_per_molecule=4,
        line_emissions_per_molecule=3.792,
        line_scan_type='descanned',
        line_num_orientations=6)
##    psfs['moderate'] = psf_comparison_pair(
##        point_resolution_improvement=3,
##        line_resolution_improvement=5.946,
##        point_emissions_per_molecule=4,
##        line_emissions_per_molecule=5.03,
##        line_scan_type='descanned',
##        line_num_orientations=8)
##    psfs['bright'] = psf_comparison_pair(
##        point_resolution_improvement=4,
##        line_resolution_improvement=7.83865,
##        point_emissions_per_molecule=4,
##        line_emissions_per_molecule=7.36,
##        line_scan_type='descanned',
##        line_num_orientations=10)
    print("Done calculating.\n")
    print("Light dose (saturation units):")
    for p in psfs:
        print(" %10s point-STED: %0.2f (excitation),  %0.2f (depletion)"%(
            p,
            psfs[p]['point_excitation_dose'],
            psfs[p]['point_depletion_dose']))
        print("%10s %i-line-STED: %0.2f (excitation),  %0.2f (depletion)"%(
            p,
            len(psfs[p]['line_sted_psfs']),
            psfs[p]['line_excitation_dose'],
            psfs[p]['line_depletion_dose']))
    result = {}
    for p in psfs.keys():
        result[p + '_point_sted'] = psfs[p]['point_sted_psf']
        result[p + '_line_%i_angles_sted'%len(psfs[p]['line_sted_psfs'])
               ] = psfs[p]['line_sted_psfs']
    return result


def psf_comparison_pair(
    point_resolution_improvement,
    line_resolution_improvement,
    point_emissions_per_molecule,
    line_emissions_per_molecule,
    line_scan_type, # 'descanned' or 'rescanned'
    line_num_orientations,
    max_excitation_brightness=0.25,
    steps_per_improved_psf_width=4, # Actual sampling, for Nyquist
    steps_per_excitation_psf_width=20, # Display sampling, for convenience
    ):
    """
    Compute a pair of PSFs, line-STED and point-STED, so we can compare
    their resolution and photodose.
    """
    # Compute correctly sampled PSFs, to get emission levels and light
    # dose:
    print("Calculating point-STED psf...")
    point = st.tune_psf(
        psf_type='point',
        scan_type='descanned',
        desired_resolution_improvement=point_resolution_improvement,
        desired_emissions_per_molecule=point_emissions_per_molecule,
        max_excitation_brightness=max_excitation_brightness,
        steps_per_improved_psf_width=steps_per_improved_psf_width,
        verbose_results=True)
    print("Calculating line-STED psf,", line_num_orientations, "orientations...")
    line = st.tune_psf(
        psf_type='line',
        scan_type=line_scan_type,
        desired_resolution_improvement=line_resolution_improvement,
        desired_emissions_per_molecule=line_emissions_per_molecule,
        max_excitation_brightness=max_excitation_brightness,
        steps_per_improved_psf_width=steps_per_improved_psf_width,
        verbose_results=True)
    # Compute finely sampled PSFs, to interpolate the PSF shapes onto a
    # consistent pixel size for display:
    fine_point = st.psf_report(
        psf_type='point',
        excitation_brightness=point['excitation_brightness'],
        depletion_brightness=point['depletion_brightness'],
        steps_per_excitation_psf_width=steps_per_excitation_psf_width,
        pulses_per_position=point['pulses_per_position'],
        verbose=False)
    fine_line = st.psf_report(
        psf_type='line',
        excitation_brightness=line['excitation_brightness'],
        depletion_brightness=line['depletion_brightness'],
        steps_per_excitation_psf_width=steps_per_excitation_psf_width,
        pulses_per_position=line['pulses_per_position'],
        verbose=False)
    # Normalize the finely sampled PSFs to give the proper emission level:
    def norm(x):
        return x / x.sum()
    point_sted_psf = [point['expected_emission'] *
                      norm(fine_point['psfs']['descan_sted'])]
    assert line['pulses_per_position'] >= line_num_orientations
    psf = {'descanned': 'descan_sted', 'rescanned': 'rescan_sted'}[line_scan_type]
    line_sted_psfs = [1 / line_num_orientations *
                      line['expected_emission'] *
                      rotate(norm(fine_line['psfs'][psf]), angle)
                      for angle in np.arange(0, 180, 180/line_num_orientations)]
    return {'point_sted_psf': point_sted_psf,
            'line_sted_psfs': line_sted_psfs,
            'point_excitation_dose': point['excitation_dose'],
            'line_excitation_dose': line['excitation_dose'],
            'point_depletion_dose': point['depletion_dose'],
            'line_depletion_dose': line['depletion_dose']}

def rotate(x, degrees):
    if degrees == 0:
        return x
    elif degrees == 90:
        return np.rot90(np.squeeze(x)).reshape(x.shape)
    else:
        return np.clip(
            interpolation.rotate(x, angle=degrees, axes=(1, 2), reshape=False),
            0, 1.1 * x.max())

if __name__ == '__main__':
    main()
