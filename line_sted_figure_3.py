import os
from subprocess import check_call
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, interpolation
import np_tif

#TODO: Nondescan multipoint doesn't seem to be assigning pixels to the
#exact right place

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
    fig = plt.figure(figsize=(10, 4), dpi=100)
    psf_width = 25 # Same as Figure 2, in object pixels
    for im_name in ('lines', 'rings', ):
        obj = np_tif.tif_to_array('test_object_'+ im_name +'.tif') / 255 + 1e-6
        for fov_size in (1, 2,):
            obj = np.tile(obj, (1, fov_size, fov_size))
            print("\nTest object:", im_name, obj.shape, obj.dtype)
            for R, num_orientations in ((1, 2), (2, 4)):
                for imaging_type in ('descan_point',
                                     'nondescan_multipoint',
                                     'descan_line',
                                     'rescan_line',):
                    comparison_name = ('%s_%ix%iFOV_%0.2fxR'%(
                        im_name, fov_size, fov_size, R)).replace('.', 'p')
                    simulate_imaging(obj,
                                     imaging_type,
                                     psf_width,
                                     R,
                                     num_orientations,
                                     pulses_per_position=1,
                                     pad=int(0.5*max(obj.shape)),
                                     comparison_name=comparison_name)

def simulate_imaging(
    obj,
    imaging_type,
    psf_width, # FWHM of diffraction limited PSFs
    R, # Excitation PSF improvement due to STED
    num_orientations, # For line STED only
    pulses_per_position, # Effectively, dwell time
    pad, # To avoid artifacts from edge effects
    comparison_name,
    ):
    """Simulate a variety of STED imaging techniques
    """
    assert len(obj.shape) == 3 and obj.shape[0] == 1
    assert imaging_type in ('descan_point', 'nondescan_multipoint',
                            'descan_line', 'rescan_line')
    assert psf_width >= 1
    psf_sigma = psf_width / (2*np.sqrt(2*np.log(2)))
    assert R >= 1
    step = int(np.round(psf_width / (4 * R))) # 4 scan positions per STED PSF
    assert num_orientations >= 1 and int(num_orientations) == num_orientations
    assert pad > 0 and int(pad) == pad
    filenames = []
    # Pad the object with zeros
    _, n_y, n_x = obj.shape
    obj = np.pad(obj, ((0, 0), (pad, pad), (pad, pad)), 'constant')
    # Excitation is either line or point
    centered_exc = np.zeros(obj.shape)
    if imaging_type in ('descan_line', 'rescan_line'):
        centered_exc[0, obj.shape[1]//2, :] = 1
        sted_sigma = (0, psf_sigma/R, 0)
        scan_positions = [(y, 0) for y in np.arange(-n_y//2, n_y//2+1, step)]
    elif imaging_type == 'descan_point':
        centered_exc[0, obj.shape[1]//2, obj.shape[2]//2] = 1
        sted_sigma = (0, psf_sigma/R, psf_sigma/R)
        scan_positions = [(y, x)
                          for y in np.arange(-n_y//2, n_y//2+1, step)
                          for x in np.arange(-n_x//2, n_x//2+1, step)]
        num_orientations = 1 # Ignore num_orientations
    else:
        assert imaging_type == 'nondescan_multipoint'
        # It's nice if the spot separation is an integer multiple of the
        # step size. It's also nice if the spot separation is a bit
        # bigger than the psf width, to minimize crosstalk.
        exc_sep = int(step * np.round(psf_width * 1.4 / step))
        centered_exc[0,  pad:-pad:exc_sep, pad:-pad:exc_sep] = 1
        sted_sigma = (0, psf_sigma/R, psf_sigma/R)
        scan_positions = [(y, x)
                          for y in np.arange(0, exc_sep, step)
                          for x in np.arange(0, exc_sep, step)]
        num_orientations = 1 # Ignore num_orientations
    centered_exc = gaussian_filter(centered_exc, sted_sigma, truncate=8)
    max_exc = centered_exc[0, pad:-pad, pad:-pad].max()
    max_glow, max_inst_sig, max_cum_sig, max_reconst, max_new_sig = 0, 0, 0, 0, 0
    for which_run in ('find_maxima', 'generate_figures'):
        camera_exposures, pulses_delivered = 0, 0
        # Rotate the object (possibly by zero degrees)
        for rot in np.arange(0, 180, 180/num_orientations)[::-1]:
            if which_run == 'find_maxima' and rot > 0: continue
            print("Orientation:", rot, "degrees")
            rot_obj = rotate(obj, rot)
            cum_detector_sig = np.zeros(obj.shape)
            reconstruction = np.zeros(obj.shape)
            # Scan the excitation to each position
            for which_pos, (shift_y, shift_x) in enumerate(scan_positions):
                pulses_delivered += pulses_per_position
                last_reconstruction = reconstruction.copy()
                exc = shift(centered_exc, (0, shift_y, shift_x))
                glow = rot_obj * exc
                descanned_glow = shift(glow, (0, -shift_y, -shift_x))
                # Compute de-rotated excitation, glow, and descanned signal
                rot_exc = rotate(exc, -rot)
                rot_glow = rotate(glow, -rot)
                rot_descanned_glow = rotate(descanned_glow, -rot)
                # Each method detects signal differently
                if imaging_type in ('descan_line', 'descan_point'):
                    # Blur the descanned glow onto the detector
                    inst_detector_sig = gaussian_filter(descanned_glow, psf_sigma)
                    cum_detector_sig = inst_detector_sig
                    if imaging_type == 'descan_line':
                        # Sum the detector pixels vertically to produce one
                        # line of the image
                        reconstruction[
                            0, shift_y+n_y//2+pad:shift_y+n_y//2+pad+step, :
                            ] = inst_detector_sig.sum(axis=1, keepdims=True)
                        camera_exposures += 1
                    elif imaging_type == 'descan_point':
                        # Sum all the detected signal to produce one pixel
                        # of the image
                        reconstruction[
                            0,
                            shift_y+n_y//2+pad:shift_y+n_y//2+pad+step,
                            shift_x+n_x//2+pad:shift_x+n_x//2+pad+step
                            ] = inst_detector_sig.sum()
                        camera_exposures = 'N/A'
                elif imaging_type == 'nondescan_multipoint':
                    # Blur the nondescanned glow onto the detector
                    inst_detector_sig = gaussian_filter(glow, psf_sigma)
                    cum_detector_sig = inst_detector_sig
                    # A region of the detector centered on each excitation
                    # point contributes to a single reconstruction pixel:
                    for j in range(int(obj.shape[-2]/exc_sep)): 
                        for i in range(int(obj.shape[-1]/exc_sep)):
                            signal_region = inst_detector_sig[
                                0,
                                max(shift_y+j*exc_sep-exc_sep//3, 0)
                                   :shift_y+j*exc_sep+exc_sep//3,
                                max(shift_x+i*exc_sep-exc_sep//3, 0)
                                   :shift_x+i*exc_sep+exc_sep//3]
                            reconstruction[ #Target pixel
                                0,
                                shift_y+j*exc_sep:
                                shift_y+j*exc_sep+step,
                                shift_x+i*exc_sep:
                                shift_x+i*exc_sep+step
                                ] = signal_region.sum()
                    camera_exposures += 1
                elif imaging_type == 'rescan_line':
                    # Blur the descanned glow, shrink it by the rescan
                    # factor.
                    scaled_descanned_sig = scale_y(
                        gaussian_filter(descanned_glow, psf_sigma),
                        scaling_factor=1/(R**2 + 1))
                    # Then rescan onto the detector, centered on the excitation
                    inst_detector_sig = shift(
                        scaled_descanned_sig, (0, shift_y, shift_x))
                    cum_detector_sig += inst_detector_sig
                    # The sensor is only read out once per orientation, at
                    # the end of the scan:
                    if (shift_y, shift_x) == scan_positions[-1]:
                        reconstruction = cum_detector_sig
                        camera_exposures += 1
                # Highlight places where the reconstruction changed
                new_signal = reconstruction - last_reconstruction
                # On the first run through, keep track of maxima to
                # allow proper display scaling:
                if which_run == 'find_maxima':
                    if which_pos == 0:
                        print("Calculating maxima for display scaling...")
                    max_glow = max(glow.max(), max_glow)
                    max_inst_sig = max(inst_detector_sig.max(), max_inst_sig)
                    max_cum_sig = max(cum_detector_sig.max(), max_cum_sig)
                    max_reconst = max(reconstruction.max(), max_reconst)
                    max_new_sig = max(new_signal.max(), max_new_sig)
                # On the second run through, we know how big images will
                # get, so we can scale correctly, so we can generate
                # figures:
                elif which_run == 'generate_figures':
                    if which_pos == 0:
                        print("Generating figures...")
                    # Try to make about 100 frames, but be careful not
                    # to skip the last one
                    num_to_skip = max(int(np.round(len(scan_positions) / 100)), 1)
                    if (which_pos % num_to_skip != 0 and
                        which_pos != len(scan_positions) - 1):
                        continue # No figure, this time.
                    filenames.append(os.path.join(
                        os.getcwd(), 'Figure_3_output',
                        imaging_type + '_%03ideg_'%rot +
                        comparison_name + '_%06i.svg'%which_pos))
                    if which_pos == len(scan_positions) -1:
                        for i in range(5): # Repeat the last frame several times
                            filenames.append(filenames[-1])
                    generate_figure(
                        filenames[-1],
                        obj[0, pad:-pad, pad:-pad] / obj.max(),
                        rot_exc[0, pad:-pad, pad:-pad] / max_exc,
                        rot_glow[0, pad:-pad, pad:-pad] / max_glow,
                        inst_detector_sig[0, pad:-pad, pad:-pad] / max_inst_sig,
                        cum_detector_sig[0, pad:-pad, pad:-pad] / max_cum_sig,
                        new_signal[0, pad:-pad, pad:-pad] / max_new_sig,
                        reconstruction[0, pad:-pad, pad:-pad] / max_reconst,
                        pulses_delivered,
                        camera_exposures)
    animate(input_filenames=filenames,
            output_filename=imaging_type + '_%03ideg_'%rot + comparison_name)
                            
def generate_figure(
    filename,
    obj,
    excitation,
    glow,
    instantaneous_detector_signal,
    cumulative_detector_signal,
    new_signal,
    reconstruction,
    pulses_delivered,
    camera_exposures,
    ):
    if camera_exposures != 'N/A':
        camera_exposures = '%04i'%camera_exposures
    print("Saving:", os.path.basename(filename))
    plt.clf()
    # Sample
    ax = plt.subplot(1, 3, 1)
    plt.title("(a) Sample")
    plt.imshow(obj, interpolation='nearest', cmap=plt.cm.gray, 
               vmin=0, vmax=obj.max())
    img = np.zeros((obj.shape[-2], obj.shape[-1], 4))
    img[:, :, 2] = excitation / 2 # Blue excitation
    img[:, :, 1] =  glow # Green glow
    img[:, :, 3] = np.sqrt(excitation) # Locally opaque
    plt.imshow(img, interpolation='nearest')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_xlabel("(d) Pulses delivered: %05i"%(pulses_delivered))
    # Detector
    ax = plt.subplot(1, 3, 2)
    plt.title("(b) Detector")
    plt.imshow(cumulative_detector_signal, interpolation='nearest',
               cmap=plt.cm.gray, vmin=0, vmax=1)
    img = np.zeros((obj.shape[-2], obj.shape[-1], 4))
    img[:, :, 1] = (instantaneous_detector_signal)
    img[:, :, 3] = np.sqrt(img[:, :, 1])
    plt.imshow(img, interpolation='nearest')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_xlabel("(e) Camera exposures: " + camera_exposures)
    # Reconstructed image
    ax = plt.subplot(1, 3, 3)
    plt.title("(c) Pre-deconvolution image(s)")
    plt.imshow(reconstruction, interpolation='nearest', cmap=plt.cm.gray,
               vmin=0, vmax=1)
    img = np.zeros((obj.shape[-2], obj.shape[-1], 4))
    img[:, :, 1] = new_signal
    img[:, :, 3] = np.sqrt(new_signal)
    plt.imshow(img, interpolation='nearest')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.subplots_adjust(left=0, bottom=0.2, right=1, top=1,
                        wspace=0, hspace=0)
    plt.savefig(filename, bbox_inches='tight', dpi=100)

def animate(input_filenames, output_filename):
    """
    I didn't find a slick way to produce gif or mp4 in Python, so I just
    farm this out to ImageMagick and ffmpeg via the command line. This
    is unlikely to work the exact same way on your system; go look at
    the docs for ImageMagick or ffmpeg if you're trying to debug this
    part.
    """
    output_filename = os.path.join(
        os.getcwd(), 'Figure_3_output', '1_Figures', output_filename)
    print("\nConverting:", output_filename)
    convert_command = ["convert", "-loop", "0"]
    for f in input_filenames:
        convert_command.extend(["-delay", "10", f])
    convert_command.extend(["-delay", "500", input_filenames[-1]])
    convert_command.append(output_filename + '.gif')
    try: # Try to convert the SVG output to animated GIF
        with open('conversion_errors.txt', 'wb') as f:
            check_call(convert_command, stderr=f, stdout=f)
    except: # Don't expect this conversion to work on anyone else's system.
        print("Gif conversion failed. Is ImageMagick installed?")
        return None
    convert_command = [
        'ffmpeg', '-y', '-i', output_filename+'.gif', output_filename+'.mp4']
    try: # Now try to convert the gif to mp4
        with open('conversion_errors.txt', 'wb') as f:
            check_call(convert_command, stderr=f, stdout=f)
            os.remove('conversion_errors.txt')
    except: # This is also unlikely to be platform independent :D
        print("MP4 conversion failed. Is ffmpeg installed?")
    return None

def rotate(x, angle_degrees):
    # scipy.interpolation.rotate has a lot of options, but I only use
    # one set of them. Also the interpolation sometimes produces small
    # negative values that it's helpful to clip off.
    if angle_degrees == 0:
        return x.copy()
    return np.clip(
        interpolation.rotate(
            x, angle=angle_degrees, axes=(1, 2), mode='nearest', reshape=False),
        0, 1.1*x.max())

def shift(x, shift):
    return np.clip(interpolation.shift(x, shift), 0, 1.1*x.max())

def scale_y(x, scaling_factor):
    assert len(x.shape) == 3 and x.shape[0] == 1 and x.shape[1] > 1
    assert float(scaling_factor) == scaling_factor
    original_shape = x.shape
    scaled = interpolation.zoom(x[0, :, :], zoom=(scaling_factor, 1))
    y_dif = original_shape[-2] - scaled.shape[-2]
    padded_x = np.pad(scaled, ((y_dif//2, y_dif-y_dif//2), (0, 0)), 'constant')
    return padded_x.reshape(original_shape)

main()


##def main():
##    if not os.path.exists('Figure_3_output'): os.mkdir('Figure_3_output')
##    if not os.path.exists('Figure_3_output/1_Figures'):
##        os.mkdir('Figure_3_output/1_Figures')
##
##    psf_width = 25 # Same as Figure 2, in object pixels
##    comparisons = {}
##    comparisons['1p0x'] = psf_comparison_pair(
##        point_resolution_improvement=0.97,
##        line_resolution_improvement=0.97,
##        point_emissions_per_molecule=3,
##        line_emissions_per_molecule=3,
##        line_scan_type='descanned',
##        line_num_orientations=4,
##        steps_per_excitation_psf_width=psf_width)
####    comparisons['2p0x'] = psf_comparison_pair(
####        point_resolution_improvement=2.,
####        line_resolution_improvement=2.,
####        point_emissions_per_molecule=4,
####        line_emissions_per_molecule=4,
####        line_scan_type='descanned',
####        line_num_orientations=4,
####        steps_per_excitation_psf_width=psf_width)
##    
##    fig = plt.figure(figsize=(10, 4), dpi=100)
##    for im_name in ('rings',):# 'lines'):
##        obj = np_tif.tif_to_array('test_object_'+ im_name +'.tif') / 255 + 1e-5
##        for fov_size in (1,):# 2,):
##            obj = np.tile(obj, (1, fov_size, fov_size))
##            print("\nTest object:", im_name, obj.shape, obj.dtype)
##            for which_comp in ('1p0x',):
####                descanned_single_point_figure(
####                    obj, psf_width, comparisons, which_comp, im_name, fov_size)
####                nondescanned_multi_point_figure(
####                    obj, psf_width, comparisons, which_comp, im_name, fov_size)
####                descanned_line_figure(
####                    obj, psf_width, comparisons, which_comp, im_name, fov_size)
##                rescanned_line_figure(
##                    obj, psf_width, comparisons, which_comp, im_name, fov_size)


##def descanned_single_point_figure(
##    obj, psf_width, comparisons, which_comp, im_name, fov_size):
##    print("\nGenerating animation for descanned point-STED")
##    print('Using parameters from comparison:', which_comp)
##    psf_sigma = psf_width / (2*np.sqrt(2*np.log(2)))
##    c = comparisons[which_comp]
##    point_R = c['point']['resolution_improvement_descanned']
##    point_step =  int(psf_width / c['point']['steps_per_excitation_psf_width'])
##    point_pulses = c['point']['pulses_per_position']
##    print("Point: R: %0.2f, Step size: %0.2f, Pulses/position: %i"%(
##        point_R, point_step, point_pulses))
##    centered_excitation = np.zeros(obj.shape)
##    centered_excitation[0, obj.shape[1]//2, obj.shape[2]//2] = 1
##    centered_excitation = gaussian_filter(
##        centered_excitation, sigma=psf_sigma/point_R, truncate=6)
##    centered_excitation /= centered_excitation.max()
##    reconstruction, contribution = np.zeros(obj.shape), np.zeros(obj.shape)
##    which_im, pulses_delivered = -1, 0
##    descan_point_filenames = []
##    print("Scanning row: ", end='')
##    for y in np.arange(-obj.shape[1]//2, obj.shape[1]//2+1, point_step):
##        print(int(y), ', ', sep='', end='')
##        y_scanned_excitation = np.roll(centered_excitation, int(y), axis=1)
##        for x in np.arange(-obj.shape[2]//2, obj.shape[2]//2+1, point_step):
##            excitation = np.roll(y_scanned_excitation, int(x), axis=2)
##            instantaneous_detector_signal = gaussian_filter(
##                np.roll(np.roll(obj * excitation,
##                                -int(y), axis=1), -int(x), axis=2),
##                sigma=psf_sigma)
##            cumulative_detector_signal = instantaneous_detector_signal
##            reconstruction[0,
##                           y+obj.shape[1]//2:y+obj.shape[1]//2+point_step,
##                           x+obj.shape[2]//2:x+obj.shape[2]//2+point_step
##                           ] += instantaneous_detector_signal.sum()
##            contribution.fill(0)
##            contribution[0,
##                         y+obj.shape[1]//2:y+obj.shape[1]//2+point_step,
##                         x+obj.shape[2]//2:x+obj.shape[2]//2+point_step
##                         ] = 1
##            pulses_delivered += point_pulses
##            which_im += 1
##            if which_im % 5 == 0:
##                descan_point_filenames.append(os.path.join(
##                    os.getcwd(), 'Figure_3_output', 'descan_point_' +
##                    which_comp + '_%ix%i_'%(fov_size, fov_size) +
##                    im_name + '_%06i.svg'%which_im))
##                generate_figure(
##                    descan_point_filenames[-1], obj, excitation,
##                    instantaneous_detector_signal, cumulative_detector_signal,
##                    contribution, reconstruction,
##                    pulses_delivered, camera_exposures='N/A')
##    descan_point_filenames.append(os.path.join(
##        os.getcwd(), 'Figure_3_output', 'descan_point_' +
##        which_comp + '_%ix%i_'%(fov_size, fov_size) +
##        im_name + '_%06i.svg'%which_im))
##    generate_figure(
##        descan_point_filenames[-1], obj, excitation,
##        instantaneous_detector_signal, cumulative_detector_signal,
##        contribution, reconstruction,
##        pulses_delivered, camera_exposures='N/A')
##    animate(input_filenames=descan_point_filenames,
##            output_filename=('descan_point_' + which_comp + '_' + im_name +
##                             '_%ix%i'%(fov_size, fov_size)))
##
##def nondescanned_multi_point_figure(
##    obj, psf_width, comparisons, which_comp, im_name, fov_size):
##    print("\nGenerating animation for nondescanned multipoint-STED")
##    print('Using parameters from comparison:', which_comp)
##    psf_sigma = psf_width / (2*np.sqrt(2*np.log(2)))
##    c = comparisons[which_comp]
##    point_R = c['point']['resolution_improvement_descanned']
##    point_step =  int(psf_width / c['point']['steps_per_excitation_psf_width'])
##    point_pulses = c['point']['pulses_per_position']
##    print("Point: R: %0.2f, Step size: %0.2f, Pulses/position: %i"%(
##        point_R, point_step, point_pulses))
##    sep = 32
##    excitation = np.zeros((1, obj.shape[1]+2*sep, obj.shape[2]+2*sep))
##    excitation[0,  ::sep, ::sep] = 1
##    excitation = gaussian_filter(excitation, sigma=psf_sigma/point_R, truncate=6)
##    excitation /= excitation[:, sep:-sep, sep:-sep].max()
##    reconstruction, contribution = np.zeros(obj.shape), np.zeros(obj.shape)
##    which_im, pulses_delivered, camera_exposures = -1, 0, 0
##    nondescan_multipoint_filenames = []
##    print("Scanning row: ", end='')
##    for y in np.arange(0, sep, point_step):
##        print(int(y), ", ", sep='', end='')
##        y_scanned_excitation = np.roll(excitation, int(y), axis=1)
##        for x in np.arange(0, sep, point_step):
##            xy_scanned_excitation = np.roll(y_scanned_excitation, int(x), axis=2
##                                            )[:, sep:-sep, sep:-sep]
##            instantaneous_detector_signal = gaussian_filter(
##                obj * xy_scanned_excitation,
##                sigma=(0, psf_sigma, psf_sigma),
##                mode='constant')
##            cumulative_detector_signal = instantaneous_detector_signal
##            contribution.fill(0)
##            for j in range(int(reconstruction.shape[1]/sep)):
##                for i in range(int(reconstruction.shape[2]/sep)):
##                    reconstruction[
##                        0,
##                        y+j*sep:y+j*sep+point_step,
##                        x+i*sep:x+i*sep+point_step
##                        ] = instantaneous_detector_signal[
##                            0,
##                            max(y-sep//3+j*sep, 0):y+sep//3+j*sep,
##                            max(x-sep//3+i*sep, 0):x+sep//3+i*sep].sum()
##                    contribution[
##                        0,
##                        y+j*sep:y+j*sep+point_step,
##                        x+i*sep:x+i*sep+point_step] = 1
##            pulses_delivered += point_pulses
##            camera_exposures += 1
##            which_im += 1
##            if which_im % 2 == 0:
##                nondescan_multipoint_filenames.append(os.path.join(
##                    os.getcwd(), 'Figure_3_output', 'nondescan_multipoint_' +
##                    which_comp + '_%ix%i_'%(fov_size, fov_size) +
##                    im_name +'_%06i.svg'%which_im))
##                generate_figure(
##                    nondescan_multipoint_filenames[-1], obj,
##                    xy_scanned_excitation,
##                    instantaneous_detector_signal, cumulative_detector_signal,
##                    contribution, reconstruction,
##                    pulses_delivered, camera_exposures='%03i'%camera_exposures)
##    nondescan_multipoint_filenames.append(os.path.join(
##        os.getcwd(), 'Figure_3_output', 'nondescan_multipoint_' +
##        which_comp + '_%ix%i_'%(fov_size, fov_size) +
##        im_name +'_%06i.svg'%which_im))
##    generate_figure(
##        nondescan_multipoint_filenames[-1], obj,
##        xy_scanned_excitation,
##        instantaneous_detector_signal, cumulative_detector_signal,
##        contribution, reconstruction,
##        pulses_delivered, camera_exposures='%03i'%camera_exposures)
##    print()
##    animate(input_filenames=nondescan_multipoint_filenames,
##            output_filename=('nondescan_multipoint_' + which_comp + '_' +
##                             im_name + '_%ix%i'%(fov_size, fov_size)))
##
##def descanned_line_figure(
##    obj, psf_width, comparisons, which_comp, im_name, fov_size):
##    print("\nGenerating animation for descanned line-STED")
##    print('Using parameters from comparison:', which_comp)
##    psf_sigma = psf_width / (2*np.sqrt(2*np.log(2)))
##    c = comparisons[which_comp]
##    line_R = c['line']['resolution_improvement_descanned']
##    line_step = int(psf_width / c['line']['steps_per_excitation_psf_width'])
##    line_pulses = c['line']['pulses_per_position']
##    line_angles = len(c['line_sted_psfs'])
##    print("Line:  R: %0.2f, Step size: %0.2f, Pulses/position: %i, Angles: %i"%(
##        line_R, line_step, line_pulses, line_angles))
##    descan_line_filenames = []
##    pulses_delivered, camera_exposures = 0, 0
##    for rotation_deg in np.arange(0, 180, 180/line_angles)[::-1]:
##        print("Angle:", rotation_deg)
##        rot_obj = np.clip(interpolation.rotate(
##            obj, angle=rotation_deg, axes=(1, 2), mode='nearest'),
##                          0, 1.1*obj.max())
##        y_i, x_i = ((rot_obj.shape[1]-obj.shape[1])//2,
##                    (rot_obj.shape[2]-obj.shape[2])//2)
##        y_f, x_f = obj.shape[1] + y_i, obj.shape[2] + x_i
##        centered_excitation = np.zeros(rot_obj.shape)
##        centered_excitation[0, rot_obj.shape[1]//2, :] = 1
##        centered_excitation = gaussian_filter(
##            centered_excitation, sigma=(0, psf_sigma/line_R, 0), truncate=6)
##        centered_excitation /= centered_excitation.max()
##        reconstruction, contribution = np.zeros(obj.shape), np.zeros(obj.shape)
##        which_im = -1
##        print("Scanning row: ", end='')
##        for y in np.arange(-obj.shape[1]//2, obj.shape[1]//2+1, line_step):
##            print(int(y), ', ', sep='', end='')
##            excitation = np.roll(centered_excitation, int(y), axis=1)
##            instantaneous_detector_signal = gaussian_filter(np.roll(
##                rot_obj * excitation, -int(y), axis=1), sigma=psf_sigma)[
##                    :, y_i:y_f, x_i:x_f]
##            cumulative_detector_signal = instantaneous_detector_signal
##            reconstruction[
##                0:, y+obj.shape[1]//2:y+obj.shape[1]//2+line_step, :
##                ] += instantaneous_detector_signal.sum(axis=1, keepdims=True)
##            contribution.fill(0)
##            contribution[0, y+obj.shape[1]//2:y+obj.shape[1]//2+line_step, :] = 1
##            # Rotate back for display
##            excitation = np.clip(interpolation.rotate(
##                excitation, angle=-rotation_deg, axes=(1, 2), reshape=False
##                ), 0, 1.1*excitation.max())[:, y_i:y_f, x_i:x_f]
##            unrot_reconstruction = np.clip(interpolation.rotate(
##                reconstruction, angle=-rotation_deg, axes=(1, 2), reshape=False
##                ), 0, 1.1*reconstruction.max())
##            unrot_contribution = np.clip(interpolation.rotate(
##                contribution, angle=-rotation_deg, axes=(1, 2), reshape=False
##                ), 0, 1.1*contribution.max())
##            pulses_delivered += line_pulses / line_angles
##            camera_exposures += 1
##            which_im += 1
##            if which_im % 2 == 0:
##                descan_line_filenames.append(os.path.join(
##                    os.getcwd(), 'Figure_3_output',
##                    'descan_line_%03ideg_'%rotation_deg + which_comp +
##                    '_' + im_name + '_%06i.svg'%which_im))
##                generate_figure(
##                    descan_line_filenames[-1], obj, excitation,
##                    instantaneous_detector_signal, cumulative_detector_signal,
##                    unrot_contribution, unrot_reconstruction,
##                    pulses_delivered, camera_exposures='%i'%camera_exposures)
##        descan_line_filenames.append(os.path.join(
##            os.getcwd(), 'Figure_3_output',
##            'descan_line_%03ideg_'%rotation_deg + which_comp +
##            '_' + im_name + '_%06i.svg'%which_im))
##        generate_figure(
##            descan_line_filenames[-1], obj, excitation,
##            instantaneous_detector_signal, cumulative_detector_signal,
##            unrot_contribution, unrot_reconstruction,
##            pulses_delivered, camera_exposures='%i'%camera_exposures)
##        print()
##    animate(input_filenames=descan_line_filenames,
##            output_filename=('descan_line_' + which_comp + '_' + im_name +
##                             '_%ix%i'%(fov_size, fov_size)))
##
##def rescanned_line_figure(
##    obj, psf_width, comparisons, which_comp, im_name, fov_size):
##    print("\nGenerating animation for rescanned line-STED")
##    print('Using parameters from comparison:', which_comp)
##    psf_sigma = psf_width / (2*np.sqrt(2*np.log(2)))
##    c = comparisons[which_comp]
##    line_R = c['line']['resolution_improvement_descanned']
##    line_step = int(psf_width / c['line']['steps_per_excitation_psf_width'])
##    line_pulses = c['line']['pulses_per_position']
##    line_angles = len(c['line_sted_psfs'])
##    rescan_ratio = line_R**2 + 1
##    print("Line:  R: %0.2f, Step size: %0.2f, Pulses/position: %i, Angles: %i"%(
##        line_R, line_step, line_pulses, line_angles))
##    rescan_line_filenames = []
##    pulses_delivered, camera_exposures = 0, -1
##    for rotation_deg in np.arange(0, 180, 180/line_angles)[::-1]:
##        camera_exposures += 1
##        print("Angle:", rotation_deg)
##        rot_obj = np.clip(interpolation.rotate(
##            obj, angle=rotation_deg, axes=(1, 2), mode='nearest'),
##                          0, 1.1*obj.max())
##        y_i, x_i = ((rot_obj.shape[1]-obj.shape[1])//2,
##                    (rot_obj.shape[2]-obj.shape[2])//2)
##        y_f, x_f = obj.shape[1] + y_i, obj.shape[2] + x_i
##        centered_excitation = np.zeros(rot_obj.shape)
##        centered_excitation[0, rot_obj.shape[1]//2, :] = 1
##        centered_excitation = gaussian_filter(
##            centered_excitation, sigma=(0, psf_sigma/line_R, 0), truncate=6)
##        centered_excitation /= centered_excitation.max()
##        reconstruction, contribution = np.zeros(obj.shape), np.zeros(obj.shape)
##        cumulative_detector_signal = np.zeros(obj.shape)
##        which_im = -1
##        print("Scanning row: ", end='')
##        for y in np.arange(-obj.shape[1]//2, obj.shape[1]//2+1, line_step):
##            print(int(y), ', ', sep='', end='')
##            excitation = np.roll(centered_excitation, int(y), axis=1)
##            descanned_signal = gaussian_filter(np.roll(
##                rot_obj * excitation, -int(y), axis=1), sigma=psf_sigma)[
##                    :, y_i:y_f, x_i:x_f]
##            scaled_descanned_signal = interpolation.zoom(
##                descanned_signal, zoom=(1, 1/rescan_ratio, 1))
##            pad = descanned_signal.shape[1] - scaled_descanned_signal.shape[1]
##            scaled_descanned_signal = np.pad(
##                scaled_descanned_signal,
##                ((0, 0), (pad//2, pad-pad//2), (0, 0)), 'constant')
##            instantaneous_detector_signal = np.roll(
##                scaled_descanned_signal, int(y), axis=1)
##            cumulative_detector_signal += instantaneous_detector_signal
####            reconstruction[
####                0:, y+obj.shape[1]//2:y+obj.shape[1]//2+line_step, :
####                ] += instantaneous_detector_signal.sum(axis=1, keepdims=True)
####            contribution.fill(0)
####            contribution[0, y+obj.shape[1]//2:y+obj.shape[1]//2+line_step, :] = 1
##            # Rotate back for display
##            excitation = np.clip(interpolation.rotate(
##                excitation, angle=-rotation_deg, axes=(1, 2), reshape=False
##                ), 0, 1.1*excitation.max())[:, y_i:y_f, x_i:x_f]
##            unrot_reconstruction = np.clip(interpolation.rotate(
##                reconstruction, angle=-rotation_deg, axes=(1, 2), reshape=False
##                ), 0, 1.1*reconstruction.max())
##            unrot_contribution = np.clip(interpolation.rotate(
##                contribution, angle=-rotation_deg, axes=(1, 2), reshape=False
##                ), 0, 1.1*contribution.max())
##            pulses_delivered += line_pulses / line_angles
##            which_im += 1
##            if which_im % 2 == 0:
##                rescan_line_filenames.append(os.path.join(
##                    os.getcwd(), 'Figure_3_output',
##                    'rescan_line_%03ideg_'%rotation_deg + which_comp +
##                    '_' + im_name + '_%06i.svg'%which_im))
##                generate_figure(
##                    rescan_line_filenames[-1], obj, excitation,
##                    instantaneous_detector_signal, cumulative_detector_signal,
##                    unrot_contribution, unrot_reconstruction,
##                    pulses_delivered, camera_exposures='%i'%camera_exposures)
##        reconstruction = cumulative_detector_signal
##        contribution.fill(1)
##        rescan_line_filenames.append(os.path.join(
##            os.getcwd(), 'Figure_3_output',
##            'rescan_line_%03ideg_'%rotation_deg + which_comp +
##            '_' + im_name + '_%06i.svg'%which_im))
##        generate_figure(
##            rescan_line_filenames[-1], obj, excitation,
##            instantaneous_detector_signal, cumulative_detector_signal,
##            unrot_contribution, unrot_reconstruction,
##            pulses_delivered, camera_exposures='%i'%camera_exposures)
##        print()
##    animate(input_filenames=rescan_line_filenames,
##            output_filename=('rescan_line_' + which_comp + '_' + im_name +
##                             '_%ix%i'%(fov_size, fov_size)))








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
