import os
import numpy as np
from scipy.ndimage import gaussian_filter
import np_tif

"""
Illustrate descanned line-STED vs. rescanned line-STED data acquisition
process, to show how rescanning greatly improves imaging speed by
reducing camera exposures, and also slightly improves image resolution
by combining information from the excitation and emission PSFs.
"""
if not os.path.exists('Figure_2_output'):
    os.mkdir('Figure_2_output')
"""
Load a simple test object
"""
obj = np_tif.tif_to_array('test_object_1.tif').astype(np.float64)
print("Test object shape:", obj.shape, '\n')
"""
We (rather accurately) approximate the effective STED excitation as a
line with a Gaussian intensity profile:
"""
excitation_sigma = 4.
excitation = np.zeros_like(obj)
excitation[0, excitation.shape[1]//2, :] = 1
excitation = gaussian_filter(excitation, sigma=(0, excitation_sigma, 0))
np_tif.array_to_tif(excitation, 'Figure_2_output/excitation.tif')
"""
The emission PSF is fatter than the STED PSF, of course:
"""
emission_sigma = 4. * np.sqrt(2)
emission_psf = np.zeros_like(obj)
emission_psf[0, emission_psf.shape[1]//2, emission_psf.shape[2]//2] = 1
emission_psf = gaussian_filter(emission_psf, sigma=emission_sigma)
np_tif.array_to_tif(emission_psf, 'Figure_2_output/emission_psf.tif')
"""
The rescan ratio depends on the relative size of the excitation and
emission width. For example, see De Luca and Manders, doi:10.1364/BOE.4.002644
"""
rescan_ratio = ((emission_sigma / excitation_sigma)**2 + 1)
"""
Now we're going to save a bunch of images to concretely illustrate how the
rescan process differs from the descan process.
"""
reconstruction_inst = np.zeros_like(obj)
reconstruction_cumu = np.zeros_like(obj)
rescanned_signal = np.zeros((
    obj.shape[0], int(rescan_ratio * obj.shape[1]), obj.shape[2]))
rescanned_signal_cumu = np.zeros((
    obj.shape[0], int(rescan_ratio * obj.shape[1]), obj.shape[2]))
for foldername in ('1_excitation',
                   '2_object',
                   '3_glow',
                   '4_descanned_signal',
                   '5_reconstruction_inst',
                   '6_reconstruction_cumu',
                   '7_rescanned_signal',
                   '8_rescanned_signal_cumu'
                   ):
    if not os.path.exists(os.path.join('Figure_2_output', foldername)):
        os.mkdir(os.path.join('Figure_2_output', foldername))
for scan_position in range(obj.shape[1]):
    print("\rScan position:", scan_position)
    """
    Algorithm:
     * Scan the excitation, save the result. 
     * Save the object too, to simplify overlaying in ImageJ.
     * Multiply the object by excitation to give the "glow"
     * Blur the excited object with the emission PSF
     * Unroll the blurred excited object, and save it; this is what a
       camera would see in a descanned system.
     * Roll the blurred excited object to the rescan position, and save
       it; this is what a camera would see in a rescanned system.
    """
    scanned_excitation = np.roll(excitation,
                                 scan_position - obj.shape[1]//2, axis=1)
    np_tif.array_to_tif(
        scanned_excitation,
        'Figure_2_output/1_excitation/excitation_%06i.tif'%(scan_position))
    np_tif.array_to_tif(
        obj,
        'Figure_2_output/2_object/object_%06i.tif'%(scan_position))
    glow = gaussian_filter(obj * scanned_excitation, sigma=emission_sigma)
    np_tif.array_to_tif(
        glow,
        'Figure_2_output/3_glow/glow_%06i.tif'%(scan_position))
    descanned_signal = np.roll(glow, obj.shape[1]//2 - scan_position, axis=1)
    np_tif.array_to_tif(
        descanned_signal,
        'Figure_2_output/4_descanned_signal/descanned_signal_%06i.tif'%(scan_position))
    reconstruction_inst.fill(0)
    reconstruction_inst[0, scan_position, :] = descanned_signal.sum(axis=1)
    np_tif.array_to_tif(
        reconstruction_inst,
        'Figure_2_output/5_reconstruction_inst/reconstruction_inst_%06i.tif'%(
            scan_position))
    reconstruction_cumu[0, scan_position, :] = descanned_signal.sum(axis=1)
    np_tif.array_to_tif(
        reconstruction_cumu,
        'Figure_2_output/6_reconstruction_cumu/reconstruction_cumu_%06i.tif'%(
            scan_position))
    rescanned_signal.fill(0)
    rescanned_signal[0, :obj.shape[1], :] = descanned_signal
    rescanned_signal = np.roll(
        rescanned_signal,
        (int(rescan_ratio) * scan_position) -obj.shape[1] // 2,
        axis=1)
    np_tif.array_to_tif(
        rescanned_signal,
        'Figure_2_output/7_rescanned_signal/rescanned_signal_%06i.tif'%(scan_position))
    rescanned_signal_cumu += rescanned_signal
    np_tif.array_to_tif(
        rescanned_signal_cumu,
        'Figure_2_output/8_rescanned_signal_cumu/rescanned_signal_cumu_%06i.tif'%(
            scan_position))
