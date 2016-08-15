import os
from subprocess import check_call
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, interpolation
import np_tif

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
            for R in (1, 2, 3): 
                comparison_name = ('%s_%ix%iFOV_%0.2fxR'%(
                    im_name, fov_size, fov_size, R)).replace('.', 'p')
                for imaging_type in ('descan_point', 'nondescan_multipoint'):
                    simulate_imaging(obj,
                                     imaging_type,
                                     psf_width,
                                     R,
                                     num_orientations=1,
                                     pulses_per_position=1,
                                     pad=psf_width,
                                     comparison_name=comparison_name)
                for imaging_type in ('descan_line', 'rescan_line'):
                    for num_orientations in (2, 4, 6):
                        simulate_imaging(obj,
                                         imaging_type,
                                         psf_width,
                                         R,
                                         num_orientations,
                                         pulses_per_position=1,
                                         pad=int(0.45*max(obj.shape)),
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
                    for y_sp in range(pad+shift_y, pad+shift_y+n_y, exc_sep):
                        for x_sp in range(pad+shift_x, pad+shift_x+n_x, exc_sep):
                            signal_region = inst_detector_sig[
                                0,
                                max(y_sp-exc_sep//3, 0):y_sp+exc_sep//3,
                                max(x_sp-exc_sep//3, 0):x_sp+exc_sep//3]
                            reconstruction[ #Target pixel
                                0,
                                y_sp-step//2:y_sp-step//2+step,
                                x_sp-step//2:x_sp-step//2+step
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
