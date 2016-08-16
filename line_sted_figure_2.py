import os
import pickle
from subprocess import call
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import interpolation, map_coordinates, gaussian_filter
import np_tif
import line_sted_tools as st
"""
Line STED is much gentler than point STED, but has very anisotropic
resolution. To give isotropic resolution and comparable image quality to
point-STED, line-STED needs to fuse multiple images taken with different
scan directions. This module compares simulations of several operating
points to illustrate that for the same light dose, fusion of multiple
line scan directions gives higher image quality than point-STED.
"""
def main():
    output_prefix = os.path.join(os.getcwd(), 'Figure_2_output/')
    psfs, comparisons = calculate_psfs(output_prefix) # Lots going on; see below.
    for im_name in ('cat', 'astronaut', 'lines', 'rings'):
        print("\nTest image:", im_name)
        if deconvolution_is_complete(psfs, output_prefix, im_name):
            print("Using saved deconvolution results.")
        else:
            # Create a deconvolution object for each of the psfs created above
            deconvolvers = {
                name: st.Deconvolver(
                    psf, (output_prefix + im_name + '_' + name + '_'))
                for name, psf in psfs.items()}
            # Use our test object to create simulated data for each imaging
            # method and dump the data to disk:
            test_object = np_tif.tif_to_array(
                'test_object_'+ im_name +'.tif').astype(np.float64)
            for decon_object in deconvolvers.values():
                decon_object.create_data_from_object(
                    test_object, total_brightness=5e10)
                decon_object.record_data()
            # Deconvolve each of our objects, saving occasionally:
            print('Deconvolving...')
            for i, save in st.logarithmic_progress(range(2**10 + 1)):
                for decon_object in deconvolvers.values():
                    decon_object.iterate()
                    if save: decon_object.record_iteration()
            print('\nDone deconvolving.')
        create_figure(comparisons, output_prefix, im_name)

def calculate_psfs(output_prefix):
    """
    Tune a family of comparable line-STED vs. point-STED psfs.
    """
    comparison_filename = os.path.join(output_prefix, 'psf_comparisons.pkl')
    if os.path.exists(comparison_filename):
        print("Loading saved PSF comparisons...")
        comparisons = pickle.load(open(comparison_filename, 'rb'))
    else:
        comparisons = {}
        comparisons['1p0x_ld'] = psf_comparison_pair(
            point_resolution_improvement=0.99, #Juuust under 1, ensures no STED
            line_resolution_improvement=0.99,
            point_emissions_per_molecule=4,
            line_emissions_per_molecule=4,
            line_scan_type='descanned',
            line_num_orientations=1,
            max_excitation_brightness=0.01) # Without STED, no point saturating
        comparisons['1p0x_lr'] = psf_comparison_pair(
            point_resolution_improvement=0.99, #Juuust under 1, ensures no STED
            line_resolution_improvement=1.38282445,
            point_emissions_per_molecule=4,
            line_emissions_per_molecule=4,
            line_scan_type='rescanned',
            line_num_orientations=2,
            max_excitation_brightness=0.01) # Without STED, no point saturating
        comparisons['1p5x_ld'] = psf_comparison_pair(
            point_resolution_improvement=1.5,
            line_resolution_improvement=2.68125,
            point_emissions_per_molecule=4,
            line_emissions_per_molecule=2.825,
            line_scan_type='descanned',
            line_num_orientations=3)
        comparisons['1p5x_lr'] = psf_comparison_pair(
            point_resolution_improvement=1.5,
            line_resolution_improvement=2.95425,
            point_emissions_per_molecule=4,
            line_emissions_per_molecule=2.618,
            line_scan_type='rescanned',
            line_num_orientations=3)
        comparisons['2p0x_ld'] = psf_comparison_pair(
            point_resolution_improvement=2,
            line_resolution_improvement=4.04057,
            point_emissions_per_molecule=4,
            line_emissions_per_molecule=3.007,
            line_scan_type='descanned',
            line_num_orientations=4)
        comparisons['2p0x_lr'] = psf_comparison_pair(
            point_resolution_improvement=2,
            line_resolution_improvement=4.07614,
            point_emissions_per_molecule=4,
            line_emissions_per_molecule=3.0227,
            line_scan_type='rescanned',
            line_num_orientations=4)
        comparisons['2p5x_ld'] = psf_comparison_pair(
            point_resolution_improvement=2.5,
            line_resolution_improvement=5.13325,
            point_emissions_per_molecule=4,
            line_emissions_per_molecule=3.792,
            line_scan_type='descanned',
            line_num_orientations=6)
        comparisons['2p5x_lr'] = psf_comparison_pair(
            point_resolution_improvement=2.5,
            line_resolution_improvement=5.15129,
            point_emissions_per_molecule=4,
            line_emissions_per_molecule=3.8,
            line_scan_type='rescanned',
            line_num_orientations=6)
        comparisons['3p0x_ld'] = psf_comparison_pair(
            point_resolution_improvement=3,
            line_resolution_improvement=5.94563,
            point_emissions_per_molecule=4,
            line_emissions_per_molecule=5.034,
            line_scan_type='descanned',
            line_num_orientations=8)
        comparisons['3p0x_lr'] = psf_comparison_pair(
            point_resolution_improvement=3,
            line_resolution_improvement=5.95587,
            point_emissions_per_molecule=4,
            line_emissions_per_molecule=5.0385,
            line_scan_type='rescanned',
            line_num_orientations=8)
        comparisons['4p0x_ld'] = psf_comparison_pair(
            point_resolution_improvement=4,
            line_resolution_improvement=7.8386627,
            point_emissions_per_molecule=4,
            line_emissions_per_molecule=7.371,
            line_scan_type='descanned',
            line_num_orientations=10)
        comparisons['4p0x_lr'] = psf_comparison_pair(
            point_resolution_improvement=4,
            line_resolution_improvement=7.840982,
            point_emissions_per_molecule=4,
            line_emissions_per_molecule=7.37195,
            line_scan_type='rescanned',
            line_num_orientations=10)
        print("Done calculating PSFs.\n")
        if not os.path.exists(output_prefix): os.mkdir(output_prefix)
        pickle.dump(comparisons, open(comparison_filename, 'wb'))
    print("Light dose (saturation units):")
    for c in sorted(comparisons.keys()):
        print("%s   point-STED:%6s (excitation),%9s (depletion)"%(
            c,
            "%0.2f"%(comparisons[c]['point']['excitation_dose']),
            "%0.2f"%(comparisons[c]['point']['depletion_dose'])))
        print("%7s-line-STED:%6s (excitation),%9s (depletion)"%(
            c + '%3s'%('%i'%len(comparisons[c]['line_sted_psfs'])),
            "%0.2f"%(comparisons[c]['line']['excitation_dose']),
            "%0.2f"%(comparisons[c]['line']['depletion_dose'])))
    psfs = {}
    for c in comparisons.keys():
        psfs[c + '_point_sted'] = comparisons[c]['point_sted_psf']
        psfs[c + '_line_%i_angles_sted'%len(comparisons[c]['line_sted_psfs'])
               ] = comparisons[c]['line_sted_psfs']
    return psfs, comparisons

def psf_comparison_pair(
    point_resolution_improvement,
    line_resolution_improvement,
    point_emissions_per_molecule,
    line_emissions_per_molecule,
    line_scan_type, # 'descanned' or 'rescanned'
    line_num_orientations,
    max_excitation_brightness=0.25,
    steps_per_improved_psf_width=4, # Actual sampling, for Nyquist
    steps_per_excitation_psf_width=25, # Display sampling, for convenience
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
            'point': point,
            'line': line}

def deconvolution_is_complete(psfs, output_prefix, im_name):
    """Check for saved deconvolution results
    """
    estimate_filenames = [output_prefix + im_name +'_'+ name +
                          '_estimate_history.tif'
                          for name in sorted(psfs.keys())]
    ft_error_filenames = [output_prefix + im_name + '_'+ name +
                          '_estimate_FT_error_history.tif'
                          for name in sorted(psfs.keys())]
    for f in estimate_filenames + ft_error_filenames:
        if not os.path.exists(f):
            return False
    return True

def rotate(x, degrees):
    if degrees == 0:
        return x
    elif degrees == 90:
        return np.rot90(np.squeeze(x)).reshape(x.shape)
    else:
        return np.clip(
            interpolation.rotate(x, angle=degrees, axes=(1, 2), reshape=False),
            0, 1.1 * x.max())

def create_figure(comparisons, output_prefix, im_name):
    print("Constructing figure images...")
    figure_dir = os.path.join(output_prefix, "1_Figures")
    if not os.path.exists(figure_dir): os.mkdir(figure_dir)
    for c in sorted(comparisons.keys()):
        # Calculating the filenames and loading the images isn't too
        # hard but the code looks like a bucket full of crabs:
        num_angles = len(comparisons[c]['line_sted_psfs'])
        point_estimate_filename = (
            output_prefix + im_name +'_'+ c +
            '_point_sted_estimate_history.tif')
        point_estimate = np_tif.tif_to_array(
            point_estimate_filename)[-1, :, :].astype(np.float64)
        line_estimate_filename = (
            output_prefix + im_name +'_'+ c +
            '_line_%i_angles_sted_estimate_history.tif'%num_angles)
        line_estimate = np_tif.tif_to_array(
            line_estimate_filename)[-1, :, :].astype(np.float64)
        true_object_filename = (
            output_prefix + im_name + '_' + c + '_point_sted_object.tif')
        true_object = np_tif.tif_to_array(
            true_object_filename)[0, :, :].astype(np.float64)
        # Not that my "publication-quality" matplotlib figure-generating
        # code is anything like readable...
        for i in range(2):
            # Comparison of point-STED and line-STED images that use the
            # same excitation and depletion dose. Side by side, and
            # overlaid with switching.
            fig = plt.figure(figsize=(10, 4), dpi=100)
            plt.suptitle("Image comparison")
            ax = plt.subplot(1, 3, 1)
            plt.imshow(point_estimate, cmap=plt.cm.gray)
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            if c.startswith('1p0x'):
                ax.set_xlabel("(a) Point confocal")
            else:
                ax.set_xlabel("(a) Point STED, R=%0.1f"%(
                    comparisons[c]['point']['resolution_improvement_descanned']))
            ax = plt.subplot(1, 3, 2)
            plt.imshow(line_estimate, cmap=plt.cm.gray)
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            if c.startswith('1p0x'):
                ax.set_xlabel("(b) %i-line confocal with equal dose"%num_angles)
            else:
                ax.set_xlabel("(b) %i-line STED with equal dose"%num_angles)
            ax = plt.subplot(1, 3, 3)
            plt.imshow((point_estimate, line_estimate)[i], cmap=plt.cm.gray)
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            if c.startswith('1p0x'):
                ax.set_xlabel(("(c) Comparison (point confocal)",
                               "(c) Comparison (%i-line confocal)"%num_angles)[i])
            else:
                ax.set_xlabel(("(c) Comparison (point STED)",
                               "(c) Comparison (%i-line STED)"%num_angles)[i])
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1,
                                wspace=0, hspace=0)
            plt.savefig(os.path.join(
                output_prefix, 'figure_2_'+ im_name +'_'+ c + '_%i.svg'%i),
                        bbox_inches='tight')
            plt.close(fig)
        imagemagick_failure = False
        try: # Try to convert the SVG output to animated GIF
            call(["convert",
                  "-loop", "0",
                  "-delay", "100", os.path.join(
                      output_prefix, 'figure_2_'+ im_name +'_'+ c + '_0.svg'),
                  "-delay", "100", os.path.join(
                      output_prefix, 'figure_2_'+ im_name +'_'+ c + '_1.svg'),
                  os.path.join(
                      figure_dir, 'figure_2_' + im_name + '_' + c + '.gif')])
        except: # Don't expect this conversion to work on anyone else's system.
            if not imagemagick_failure:
                print("Gif conversion failed. Is ImageMagick installed?")
            imagemagick_failure = True
        # Error vs. spatial frequency for the point vs. line images
        # plotted above
        fig = plt.figure(figsize=(10, 4), dpi=100)
        def fourier_error(x):
            return (np.abs(np.fft.fftshift(np.fft.fftn(x - true_object))) /
                    np.prod(true_object.shape))
        ax = plt.subplot2grid((12, 12), (0, 0), rowspan=8, colspan=8)
        plt.imshow(np.log(1 + np.hstack((fourier_error(point_estimate),
                                         fourier_error(line_estimate)))),
                   cmap=plt.cm.gray)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        n_x, n_y = true_object.shape
        ang = (0) * 2*np.pi/360
        rad = 0.3
        x0, x1 = (0.5 + rad * np.array((-np.cos(ang), np.cos(ang)))) * n_x
        y0, y1 = (0.5 + rad * np.array((-np.sin(ang), np.sin(ang)))) * n_y
        ax.plot([x0, x1], [y0, y1], 'go-')
        ax.plot([x0 + n_x, x1 + n_x], [y0, y1], 'b+-')
        ang = (90/num_angles) * 2*np.pi/360
        x0_2, x1_2 = (0.5 + rad * np.array((-np.cos(ang), np.cos(ang)))) * n_x
        y0_2, y1_2 = (0.5 + rad * np.array((-np.sin(ang), np.sin(ang)))) * n_y
        ax.plot([x0_2 + n_x, x1_2 + n_x], [y0_2, y1_2], 'b+:')
        ax.set_xlabel("(d) Error vs. spatial frequency")
        # Error vs. spatial frequency for lines extracted from the
        # 2D fourier error plots
        ax = plt.subplot2grid((12, 12), (2, 8), rowspan=6, colspan=4)
        samps = 1000
        xy = np.vstack((np.linspace(x0, x1, samps), np.linspace(y0, y1, samps)))
        xy_2 = np.vstack((np.linspace(x0_2, x1_2, samps),
                          np.linspace(y0_2, y1_2, samps)))
        z_point = map_coordinates(np.transpose(fourier_error(point_estimate)), xy)
        z_line = map_coordinates(np.transpose(fourier_error(line_estimate)), xy)
        z_line_2 = map_coordinates(np.transpose(fourier_error(line_estimate)),
                                   xy_2)
        ax.semilogy(gaussian_filter(z_point, sigma=samps/80),
                    'g-', label='Point')
        ax.semilogy(gaussian_filter(z_line, sigma=samps/80),
                    'b-', label='Line, best')
        ax.semilogy(gaussian_filter(z_line_2, sigma=samps/80),
                    'b:', label='Line, worst')
        ax.axes.get_xaxis().set_ticks([])
        ax.set_xlabel("(e) Error vs. spatial frequency")
        plt.ylim(5e0, 9e5)
        ax.yaxis.tick_right()
        ax.grid('on')
        ax.legend(loc=(-0.1, .75), fontsize=8)
        # PSFs for point and line STED
        ax = plt.subplot2grid((12, 12), (10, 0), rowspan=2, colspan=2)
        plt.imshow(comparisons[c]['point_sted_psf'][0][0, :, :],
                   cmap=plt.cm.gray)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        for n, im in enumerate(comparisons[c]['line_sted_psfs']):
            ax = plt.subplot2grid((12, 12), (10, 2+n), rowspan=2, colspan=1)
            plt.imshow(im[0, :, :], cmap=plt.cm.gray)
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
        fig.text(x=0.16, y=0.25, s="(f) Point PSF", fontsize=10)
        fig.text(x=0.26, y=0.25, s="(g) Line PSF(s)", fontsize=10)
        # Save the results
        fig.text(x=0.65, y=0.83,
                 s=("Excitation dose: %6s\n"%(
                     "%0.1f"%(comparisons[c]['point']['excitation_dose'])) +
                    "Depletion dose: %7s"%(
                        "%0.1f"%(comparisons[c]['point']['depletion_dose']))),
                 fontsize=12, fontweight='bold')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0, hspace=0)
        plt.savefig(os.path.join(
            figure_dir, 'figure_2_'+ im_name +'_'+ c + '.svg'),
                    bbox_inches='tight')
        plt.close(fig)
    print("Done constructing figure images.")

if __name__ == '__main__':
    main()
