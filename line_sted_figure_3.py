import os
from subprocess import call
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import np_tif
from line_sted_figure_2 import calculate_psfs

"""
Illustrate descanned line-STED vs. rescanned line-STED data acquisition
process, to show how rescanning greatly improves imaging speed by
reducing camera exposures, and also slightly improves image resolution
by combining information from the excitation and emission PSFs.
"""
def main():
    if not os.path.exists('Figure_3_output'): os.mkdir('Figure_3_output')
    if not os.path.exists('Figure_3_output/1_Figures'):
        os.mkdir('Figure_3_output/1_Figures')

    # Choose which parameters to compare
    _, comparisons = calculate_psfs(output_prefix='Figure_2_output')
    which_comp = '1p0x'
    print('\nUsing parameters from comparison:', which_comp)
    c = comparisons[which_comp]
    psf_width = 25 # Same as Figure 2, in object pixels
    psf_sigma = psf_width / (2*np.sqrt(2*np.log(2)))

    point_R = c['point']['resolution_improvement_descanned']
    point_step =  int(psf_width / c['point']['steps_per_excitation_psf_width'])
    point_pulses = c['point']['pulses_per_position']
    print("Point: R: %0.2f, Step size: %0.2f, Pulses/position: %i"%(
        point_R, point_step, point_pulses))
    line_R = c['line']['resolution_improvement_descanned']
    line_step = int(psf_width / c['line']['steps_per_excitation_psf_width'])
    line_pulses = c['line']['pulses_per_position']
    line_angles = len(c['line_sted_psfs'])
    print("Line:  R: %0.2f, Step size: %0.2f, Pulses/position: %i, Angles: %i"%(
        line_R, line_step, line_pulses, line_angles))

    # Choose a test object
    im_name = 'lines'
    obj = np_tif.tif_to_array('test_object_'+ im_name +'.tif') / 255
    print("\nTest object:", im_name, obj.shape, obj.dtype)

    # Descanned single point
    centered_excitation = np.zeros(obj.shape)
    centered_excitation[0, obj.shape[1]//2, obj.shape[2]//2] = 1
    centered_excitation = gaussian_filter(centered_excitation,
                                          sigma=psf_sigma/point_R)
    centered_excitation /= centered_excitation.max()
    reconstruction, contribution = np.zeros(obj.shape), np.zeros(obj.shape)
    which_im, pulses_delivered, intensities_measured = -1, 0, 0
    descan_point_filenames = []
    fig = plt.figure(figsize=(10, 4), dpi=100)
    print("\nGenerating animation for descanned point-STED")
    print("Scanning row: ", end='')
    for y in np.arange(-obj.shape[1]//2, obj.shape[1]//2+1, point_step):
        print(int(y), ', ', sep='', end='')
        y_scanned_excitation = np.roll(centered_excitation, int(y), axis=1)
        for x in np.arange(-obj.shape[2]//2, obj.shape[2]//2+1, point_step):
            excitation = np.roll(y_scanned_excitation, int(x), axis=2)
            instantaneous_detector_signal = gaussian_filter(
                np.roll(np.roll(obj * excitation,
                                -int(y), axis=1), -int(x), axis=2),
                sigma=psf_sigma)
            cumulative_detector_signal = instantaneous_detector_signal
            reconstruction[0,
                           y+obj.shape[1]//2:y+obj.shape[1]//2+point_step,
                           x+obj.shape[2]//2:x+obj.shape[2]//2+point_step
                           ] += instantaneous_detector_signal.sum()
            contribution.fill(0)
            contribution[0,
                         y+obj.shape[1]//2:y+obj.shape[1]//2+point_step,
                         x+obj.shape[2]//2:x+obj.shape[2]//2+point_step
                         ] = 1
            pulses_delivered += point_pulses
            intensities_measured += 1
            which_im += 1
            if which_im % 15 != 0: continue
            descan_point_filenames.append(os.path.join(
                os.getcwd(), 'Figure_3_output',
                'descanned_point_'+which_comp+'_'+im_name+'_%06i.svg'%which_im))
            generate_figure(
                descan_point_filenames[-1], obj, excitation,
                10*instantaneous_detector_signal, cumulative_detector_signal,
                contribution, reconstruction,
                pulses_delivered, intensities_measured)
    animate(input_filenames=descan_point_filenames,
            output_filename='descan_point_'+which_comp+'.gif')
    animate(input_filenames=descan_point_filenames,
            output_filename='descan_point_'+which_comp+'.mp4')

    # Nondescanned multi-point
    sep = 32
    excitation = np.zeros((1, obj.shape[1]+2*sep, obj.shape[2]+2*sep))
    excitation[0,  ::sep, ::sep] = 1
    excitation = gaussian_filter(excitation, sigma=psf_sigma/point_R)
    excitation /= excitation[:, sep:-sep, sep:-sep].max()
    reconstruction, contribution = np.zeros(obj.shape), np.zeros(obj.shape)
    which_im, pulses_delivered, intensities_measured = -1, 0, 0
    nondescan_multipoint_filenames = []
    fig = plt.figure(figsize=(10, 4), dpi=100)
    print("\nGenerating animation for nondescanned multipoint-STED")
    print("Scanning row: ", end='')
    for y in np.arange(0, sep, point_step):
        print(int(y), ", ", sep='', end='')
        y_scanned_excitation = np.roll(excitation, int(y), axis=1)
        for x in np.arange(0, sep, point_step):
            xy_scanned_excitation = np.roll(y_scanned_excitation, int(x), axis=2
                                            )[:, sep:-sep, sep:-sep]
            instantaneous_detector_signal = gaussian_filter(
                obj * xy_scanned_excitation,
                sigma=(0, psf_sigma, psf_sigma),
                mode='constant')
            cumulative_detector_signal = instantaneous_detector_signal
            contribution.fill(0)
            for j in range(int(reconstruction.shape[1]/sep)):
                for i in range(int(reconstruction.shape[2]/sep)):
                    reconstruction[
                        0,
                        y+j*sep:y+j*sep+point_step,
                        x+i*sep:x+i*sep+point_step
                        ] += instantaneous_detector_signal[
                            0,
                            max(y-sep//3+j*sep, 0):y+sep//3+j*sep,
                            max(x-sep//3+i*sep, 0):x+sep//3+i*sep].sum()
                    contribution[
                        0,
                        y+j*sep:y+j*sep+point_step,
                        x+i*sep:x+i*sep+point_step] = 1
            pulses_delivered += point_pulses
            intensities_measured += np.prod(instantaneous_detector_signal.shape)
            which_im += 1
            if which_im % 2 != 1: continue
            nondescan_multipoint_filenames.append(os.path.join(
                os.getcwd(), 'Figure_3_output',
                'parallel_point_' + im_name +'_%i.svg'%which_im))
            generate_figure(
                nondescan_multipoint_filenames[-1], obj, xy_scanned_excitation,
                10*instantaneous_detector_signal, cumulative_detector_signal,
                contribution, reconstruction,
                pulses_delivered, intensities_measured)
    animate(input_filenames=nondescan_multipoint_filenames,
            output_filename='nondescan_multipoint_'+which_comp+'.gif')
    animate(input_filenames=nondescan_multipoint_filenames,
            output_filename='nondescan_multipoint_'+which_comp+'.mp4')

def generate_figure(
    filename,
    obj,
    excitation,
    instantaneous_detector_signal,
    cumulative_detector_signal,
    contribution,
    reconstruction,
    pulses_delivered,
    intensities_measured,
    ):
    plt.clf()
    # Sample
    ax = plt.subplot(1, 3, 1)
    plt.title("(a) Sample")
    plt.imshow(obj[0, :, :], cmap=plt.cm.gray)
    img = np.zeros((obj.shape[1], obj.shape[2], 4))
    img[:, :, 2] = excitation[0, :, :]/2 # Blue excitation
    img[:, :, 1] = (obj * excitation)[0, :, :] # Green glow
    img[:, :, 3] = excitation[0, :, :] # Locally opaque
    plt.imshow(img)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_xlabel("(d) Pulses delivered: %04i"%(pulses_delivered))
    # Detector
    ax = plt.subplot(1, 3, 2)
    plt.title("(b) Detector")
    plt.imshow(cumulative_detector_signal[0, :, :],
               cmap=plt.cm.gray, vmin=0, vmax=1)
    img = np.zeros((obj.shape[1], obj.shape[2], 4))
    img[:, :, 1] = instantaneous_detector_signal
    img[:, :, 3] = instantaneous_detector_signal > 0.01
    plt.imshow(img)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_xlabel("(e) Intensity measurements: %07i"%(intensities_measured))
    # Reconstructed image
    ax = plt.subplot(1, 3, 3)
    plt.title("(c) Reconstruction")
    plt.imshow(reconstruction[0, :, :], cmap=plt.cm.gray,
               vmin=0, vmax=reconstruction.max())
    img = np.zeros((obj.shape[1], obj.shape[2], 4))
    img[:, :, 1] = contribution
    img[:, :, 3] = contribution
    plt.imshow(img)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1,
                        wspace=0, hspace=0)
    plt.savefig(filename, bbox_inches='tight')

def animate(input_filenames, output_filename):
    """
    I didn't find a slick way to produce gif or mp4 in Python, so I just
    farm this out to ImageMagick via the command line. This is unlikely
    to work the exact same way on your system; go look at the docs for
    ImageMagick or ffmpeg if you're trying to debug this part.
    """
    output_filename = os.path.join(
        os.getcwd(), 'Figure_3_output', '1_Figures', output_filename)
    print("\nConverting:", output_filename)
    convert_command = ["convert", "-loop", "0"]
    for f in input_filenames:
        convert_command.extend(["-delay", "10", f])
    convert_command.extend(["-delay", "500", input_filenames[-1]])
    convert_command.append(output_filename)
    try: # Try to convert the SVG output to animated GIF
        call(convert_command)
    except: # Don't expect this conversion to work on anyone else's system.
        print("Gif conversion failed. Is ImageMagick installed?")

main()











##"""
##The rescan ratio depends on the relative size of the excitation and
##emission width. For example, see De Luca and Manders, doi:10.1364/BOE.4.002644
##"""
##rescan_ratio = ((emission_sigma / excitation_sigma)**2 + 1)
##
##reconstruction_inst = np.zeros_like(obj)
##reconstruction_cumu = np.zeros_like(obj)
##rescanned_signal = np.zeros((
##    obj.shape[0], int(rescan_ratio * obj.shape[1]), obj.shape[2]))
##rescanned_signal_cumu = np.zeros((
##    obj.shape[0], int(rescan_ratio * obj.shape[1]), obj.shape[2]))
##for scan_position in range(obj.shape[1]):
##    print("\rScan position:", scan_position)
##    """
##    Algorithm:
##     * Scan the excitation, save the result. 
##     * Save the object too, to simplify overlaying in ImageJ.
##     * Multiply the object by excitation to give the "glow"
##     * Blur the excited object with the emission PSF
##     * Unroll the blurred excited object, and save it; this is what a
##       camera would see in a descanned system.
##     * Roll the blurred excited object to the rescan position, and save
##       it; this is what a camera would see in a rescanned system.
##    """
##    scanned_excitation = np.roll(excitation,
##                                 scan_position - obj.shape[1]//2, axis=1)
##    np_tif.array_to_tif(
##        scanned_excitation,
##        'Figure_2_output/1_excitation/excitation_%06i.tif'%(scan_position))
##    np_tif.array_to_tif(
##        obj,
##        'Figure_2_output/2_object/object_%06i.tif'%(scan_position))
##    glow = gaussian_filter(obj * scanned_excitation, sigma=emission_sigma)
##    np_tif.array_to_tif(
##        glow,
##        'Figure_2_output/3_glow/glow_%06i.tif'%(scan_position))
##    descanned_signal = np.roll(glow, obj.shape[1]//2 - scan_position, axis=1)
##    np_tif.array_to_tif(
##        descanned_signal,
##        'Figure_2_output/4_descanned_signal/descanned_signal_%06i.tif'%(scan_position))
##    reconstruction_inst.fill(0)
##    reconstruction_inst[0, scan_position, :] = descanned_signal.sum(axis=1)
##    np_tif.array_to_tif(
##        reconstruction_inst,
##        'Figure_2_output/5_reconstruction_inst/reconstruction_inst_%06i.tif'%(
##            scan_position))
##    reconstruction_cumu[0, scan_position, :] = descanned_signal.sum(axis=1)
##    np_tif.array_to_tif(
##        reconstruction_cumu,
##        'Figure_2_output/6_reconstruction_cumu/reconstruction_cumu_%06i.tif'%(
##            scan_position))
##    rescanned_signal.fill(0)
##    rescanned_signal[0, :obj.shape[1], :] = descanned_signal
##    rescanned_signal = np.roll(
##        rescanned_signal,
##        (int(rescan_ratio) * scan_position) -obj.shape[1] // 2,
##        axis=1)
##    np_tif.array_to_tif(
##        rescanned_signal,
##        'Figure_2_output/7_rescanned_signal/rescanned_signal_%06i.tif'%(scan_position))
##    rescanned_signal_cumu += rescanned_signal
##    np_tif.array_to_tif(
##        rescanned_signal_cumu,
##        'Figure_2_output/8_rescanned_signal_cumu/rescanned_signal_cumu_%06i.tif'%(
##            scan_position))
