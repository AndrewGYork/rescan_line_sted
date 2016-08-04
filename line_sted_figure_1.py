import os
import numpy as np
import matplotlib.pyplot as plt
from line_sted_tools import psf_report

def main():
    psf_types = ['line', 'point']
    excitation_brightnesses = [0.25, 1, 4]
    depletion_brightnesses = [0, 1,  3, 9, 27]
    sampling_rates = [4, 6, 8, 12]
    pulses_per_position = [1, 2, 4, 8]
    for exc in excitation_brightnesses:
        for dep in depletion_brightnesses:
            for samps in sampling_rates:
                for pulses in pulses_per_position:
                    for p in psf_types:
                        create_figure(
                            psf_type=p,
                            excitation_brightness=exc,
                            depletion_brightness=dep,
                            steps_per_excitation_psf_width=samps,
                            pulses_per_position=pulses)

def create_figure(
    psf_type='point', # Point or line
    excitation_brightness=0.2, # Peak brightness in saturation units
    depletion_brightness=20, # Peak brightness in saturation units
    steps_per_excitation_psf_width=12, # Small? Bad res. Big? Bad dose.
    pulses_per_position=1, # Think of this as "dwell time"
    ):
    args = locals()
    args['verbose'] = True
    args['output_dir'] = None
    ###################
    # Calculations
    ###################
    # Calculate the relevant PSFs and light dosages:
    print("Generating figure with parameters:")
    report = psf_report(**args)
    # Also calculate the same PSFs on a much finer grid, so we can show
    # band limits and sampling correctly:
    fine_args = dict(args)
    fine_args['steps_per_excitation_psf_width'] = 30
    fine_args['verbose'] = False
    fine_report = psf_report(**fine_args)
    fine_excitation = fine_report['psfs']['excitation'][0, :, :]
    fine_depletion = fine_report['psfs']['depletion'][0, :, :]
    fine_excitation_frac = fine_report['psfs']['excitation_fraction'][0, :, :]
    fine_depletion_frac = fine_report['psfs']['depletion_fraction'][0, :, :]
    fine_sted = fine_report['psfs']['sted'][0, :, :]
    # Calculate the pixel positions of the samples:
    step_size_ratio = (fine_args['steps_per_excitation_psf_width'] /
                       args['steps_per_excitation_psf_width'])
    if psf_type == 'point':
        y_vals = np.arange(0, fine_excitation.shape[1], step_size_ratio)
    elif psf_type == 'line':
        y_vals = [fine_excitation.shape[1] // 2]
    sample_points_x, sample_points_y = [], []
    for x in np.arange(0, fine_excitation.shape[0], step_size_ratio):
        for y in y_vals:
            sample_points_x.append(x)
            sample_points_y.append(y)
    ###################
    # Plotting
    ###################
    fig = plt.figure(figsize=(20, 5), dpi=100)
    fig.text(x=0.23, y=0.02, s="(d)", fontsize=15)
    fig.text(x=0.25, y=-0.01, s='' + 
        "Excitation dose:%8s\n"%(
            '%0.2f'%report['excitation_dose']) +
        " Depletion dose:%8s"%(
            '%0.2f'%report['depletion_dose']),
        fontweight=800, multialignment='left', family='monospace')
    fig.text(x=0.41, y=0.02, s="(e)", fontsize=15)
    fig.text(x=0.43, y=-0.01, s='' +
        " Emissions:%5s per molecule\n"%(
            '%0.2f'%report['expected_emission']) +
        "Resolution:%5sx diffraction"%(
            '%0.1f'%report['resolution_improvement_descanned']),
        fontweight=800, multialignment='left', family='monospace')
    #####
    # Plot a 1D lineout of the excitation and depletion illumination,
    # with sample points overlayed:
    #####
    ax = plt.subplot(2, 3, 1)
    plt.title("(a) Illumination brightness")
    # Excitation
    ax.plot(fine_excitation[fine_excitation.shape[0]//2, :],
            c='blue', linestyle=':', linewidth=3)
    for t in ax.get_yticklabels():
        t.set_color('blue')
    ax.axes.get_xaxis().set_ticks([])
    ax.set_ylabel('Excitation fluence per pulse', color='blue')
    plt.ylim(0, fine_excitation[fine_excitation.shape[0]//2, :].max())
    # Sample points overlayed
    ax.set_xlabel('Scan position')
    for x in set(sample_points_x):
        ax.axvline(x, ymin=0, ymax=0.1, c='gray')
    # Depletion
    ax2 = ax.twinx()
    ax2.plot(fine_depletion[fine_depletion.shape[0]//2, :], color='green')
    ax2.set_ylabel('Depletion fluence per pulse', color='green')
    plt.ylim(0, max(fine_depletion[fine_depletion.shape[0]//2, :].max(), 0.1))
    for t in ax2.get_yticklabels():
        t.set_color('green')
    #####
    # Plot a 2D color version of the excitation and depletion
    # illumination, with the sample points overlayed:
    #####
    ax = plt.subplot(2, 3, 4)
    def norm(x):
        if x.max() == x.min(): return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())
    rgb = np.zeros(fine_excitation.shape + (3,))
    rgb[:, :, 2] = norm(fine_excitation)
    rgb[:, :, 1] = norm(fine_depletion)
    ax.imshow(rgb, interpolation='nearest')
    ax.scatter(sample_points_x, sample_points_y, c='gray', s=1)
    plt.xlim(0, rgb.shape[0])
    plt.ylim(rgb.shape[1] * 0.17, rgb.shape[1]*0.83)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    #####
    # Plot a 1D lineout of the excitation and depletion probabilities,
    # with sample points overlayed:
    #####
    ax = plt.subplot(2, 3, 2)
    plt.title("(b) Transition probability")
    # Excitation
    ax.plot(fine_excitation_frac[fine_excitation_frac.shape[0]//2, :],
            color='blue', linestyle=':', label=' Pre-depletion', linewidth=3)
    ax.plot(fine_sted[fine_sted.shape[0]//2, :],
            color='blue', label='Post-depletion', linewidth=3)
    for t in ax.get_yticklabels():
        t.set_color('blue')
    ax.axes.get_xaxis().set_ticks([])
    ax.set_ylabel('Excitation probability per pulse', color='blue')
    plt.ylim(0, 1.19)
    # Sample points overlayed
    ax.set_xlabel('Scan position')
    for x in set(sample_points_x):
        ax.axvline(x, ymin=0, ymax=0.1, c='gray')
    # Depletion
    ax2 = ax.twinx()
    ax2.plot(1 - fine_depletion_frac[fine_depletion_frac.shape[0]//2, :],
             color='green')
    ax2.set_ylabel('Depletion probability per pulse', color='green')
    plt.ylim(0, 1.19)
    for t in ax2.get_yticklabels(): t.set_color('green')
    # Shenanigans to get the legend overlay right:
    ax.set_zorder(1)
    ax.set_frame_on(False)
    ax2.set_frame_on(True)
    ax.legend(loc=(0.0, 0.855), fontsize=9, framealpha=0.9)
    #####
    # Plot a 2D color version of the excitation and depletion
    # probabilities, with the sample points overlayed:
    #####
    ax = plt.subplot(2, 3, 5)
    rgb = np.zeros(fine_excitation_frac.shape + (3,))
    rgb[:, :, 2] = norm(fine_sted)
    rgb[:, :, 1] = norm(1 - fine_depletion_frac)
    ax.imshow(rgb, interpolation='nearest')
    ax.scatter(sample_points_x, sample_points_y, c='gray', s=3)
    plt.xlim(0, rgb.shape[0])
    plt.ylim(rgb.shape[1] * 0.17, rgb.shape[1]*0.83)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    #####
    # Plot a 1D lineout of the excitation and STED OTF,
    # with the Nyquist frequency overlayed:
    #####
    ax = plt.subplot(1, 3, 3)
    plt.title("(c) Frequency performance")
    # Excitation and STED OTFs:
    def freqs(x, dc_term):
        otf = np.fft.fftshift(np.abs(np.fft.fft(x)))
        otf /= otf.max()
        otf *= dc_term
        return otf
    if psf_type == 'point':
        exc_dc_term = (report['psfs']['excitation_fraction'].sum() *
                       report['pulses_per_position'])
        sted_dc_term = (report['psfs']['sted'].sum() *
                        report['pulses_per_position'])
    elif psf_type == 'line':
        midline = report['psfs']['excitation'].shape[1] // 2
        exc_dc_term = (report['pulses_per_position'] *
                       report['psfs']['excitation_fraction'][0, midline, :].sum())
        sted_dc_term = (report['pulses_per_position'] *
                        report['psfs']['sted'][0, midline, :].sum())
    exc_otf = freqs(fine_excitation_frac[fine_excitation_frac.shape[0]//2, :],
                    dc_term=exc_dc_term)
    ax.semilogy(exc_otf, label='Excitation OTF', linewidth=3,
                linestyle=':', color='blue')
    sted_otf = freqs(fine_sted[fine_sted.shape[0]//2, :],
                     dc_term=(sted_dc_term))
    ax.semilogy(sted_otf, label='STED OTF', linewidth=3, color='blue')
    # Sampling frequency overlayed
    ax.axvline(len(exc_otf)//2 + 0.5*len(exc_otf) / step_size_ratio,
               color='red', linewidth=4, ymin=0, ymax=0.8, alpha=0.5)
    ax.axvline(len(exc_otf)//2 - 0.5*len(exc_otf) / step_size_ratio,
               color='red', linewidth=4, ymin=0, ymax=0.8, alpha=0.5,
               label="Nyquist limit")
    plt.xlim(fine_excitation_frac.shape[1]*0.25,
             fine_excitation_frac.shape[1]*0.75)
    plt.ylim(1e-2, 9e3)
    ax.set_xlabel('Spatial frequency')
    ax.set_ylabel('Transmission amplitude')
    ax.axes.get_xaxis().set_ticks([])
    ax.legend(loc=(0.04, 0.88), fontsize=9)
    #####
    # Make everything line up
    #####
    plt.subplots_adjust(left=0.25, bottom=0.0, right=0.75, top=1.0,
                        wspace=0.52, hspace=-0.1)
    ###################
    # Save the figure
    ###################
    filename = (
        psf_type + '_' +
        ('%0.2fexc'%(excitation_brightness)).replace('.', 'p') + '_' +
        ('%0.2fdep'%(depletion_brightness)).replace('.', 'p') + '_' +
        '%03isamps'%steps_per_excitation_psf_width + '_' +
        '%03ipulses'%pulses_per_position +
        '.svg')
    print("Saving:", filename, '\n')
    if not os.path.exists('Figure_1_output'):
        os.mkdir('Figure_1_output')
    plt.savefig(os.path.join('Figure_1_output', filename),
                bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    main()
