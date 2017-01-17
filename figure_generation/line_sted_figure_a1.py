#!/usr/bin/python3
# Dependencies from the Python 3 standard library:
import os
# Dependencies from the Scipy stack https://www.scipy.org/stackspec.html :
import numpy as np
import matplotlib.pyplot as plt
# Dependencies from https://github.com/AndrewGYork/rescan_line_sted :
from line_sted_tools import psf_report, tune_psf

output_directory = os.path.join(os.pardir, 'images', 'figure_a1')
if not os.path.isdir(output_directory): os.mkdir(output_directory)

def main():
    print("Plotting resolution improvement vs. depletion brightness")
    r_vs_depletion_brightness()
    print("Plotting resolution improvement vs. depletion dose")
    r_vs_depletion_dose()
    print("Plotting resolution improvement vs. depletion dose,",
          "with variable step size")
    r_vs_depletion_dose_variable_steps()
    print("Plotting resolution improvement vs. depletion dose,",
          "with variable step size and variable excitation brightness")
    r_vs_depletion_dose_variable_steps_variable_excitation()
    
def r_vs_depletion_brightness():
    """A plot of resolution improvement vs. depletion intensity, for
    fixed step size and excitation brightness, and a single line-STED
    orientation: """
    depletion_brightnesses = np.linspace(0, 18, 75)
    resolution_improvement = {'point': [],
                              'line': []}
    for depletion_brightness in depletion_brightnesses:
        for psf_type in ('point', 'line'):
            report = psf_report(
                psf_type='point',
                excitation_brightness=0.25,
                depletion_brightness=depletion_brightness,
                steps_per_excitation_psf_width=4,
                pulses_per_position=1,
                verbose=False)
            resolution_improvement[psf_type].append(
                report['resolution_improvement_descanned'])
    fig = plt.figure(figsize=(12, 4), dpi=100)
    plt.plot(depletion_brightnesses, resolution_improvement['point'],
             '.', label='Descanned point STED')
    plt.plot(depletion_brightnesses, resolution_improvement['line'],
             '-', label='Descanned line STED')
    plt.xlim(0, 17)
    plt.ylim(1, 4.5)
    plt.legend(loc='lower right')
    plt.grid()
    plt.xlabel('Peak depletion brightness (saturation units)')
    plt.ylabel('Resolution improvement (vs. diffraction limit)')
    plt.title('Resolution improvement vs. depletion brightness\n' +
              'for fixed step size and fixed excitation brightness')
    plt.savefig(os.path.join(output_directory, '1_r_vs_depletion_brightness.svg'),
                bbox_inches='tight')
    plt.close(fig)

def r_vs_depletion_dose():
    """A plot of resolution improvement vs. depletion dose, for fixed
    step size and excitation brightness, and a single line-STED
    orientation: """
    depletion_brightnesses = np.linspace(0, 18, 200)
    resolution_improvement = {'point': [],
                              'line': []}
    depletion_dose = {'point': [],
                      'line': []}
    for depletion_brightness in depletion_brightnesses:
        for psf_type in ('point', 'line'):
            report = psf_report(
                psf_type=psf_type,
                excitation_brightness=0.25,
                depletion_brightness=depletion_brightness,
                steps_per_excitation_psf_width=4,
                pulses_per_position=1,
                verbose=False)
            resolution_improvement[psf_type].append(
                report['resolution_improvement_descanned'])
            depletion_dose[psf_type].append(report['depletion_dose'])
    fig = plt.figure(figsize=(12, 4), dpi=100)
    plt.plot(depletion_dose['point'], resolution_improvement['point'],
             '.', label='Descanned point STED')
    plt.plot(depletion_dose['line'], resolution_improvement['line'],
             '-', label='Descanned line STED')
    plt.xlim(0, 750)
    plt.ylim(1, 4.5)
    plt.legend(loc='lower right')
    plt.grid()
    plt.xlabel('Depletion dose (saturation units)')
    plt.ylabel('Resolution improvement (vs. diffraction limit)')
    plt.title('Resolution improvement vs. depletion dose\n' +
              'for fixed step size and fixed excitation brightness')
    plt.savefig(os.path.join(output_directory, '2_r_vs_depletion_dose.svg'),
                bbox_inches='tight')
    plt.close(fig)

def r_vs_depletion_dose_variable_steps():
    """A plot of resolution improvement vs. depletion dose, for fixed
    excitation brightness and a single line-STED orientation, but with
    a step size which varies to keep steps per STED PSF fixed:"""
    depletion_brightnesses = np.linspace(0.1, 18, 200)
    resolution_improvement = {'point': [],
                              'line': []}
    depletion_dose = {'point': [],
                      'line': []}
    print('Calculating', end='')
    for depletion_brightness in depletion_brightnesses:
        print('.', end='', sep='')
        for psf_type in ('point', 'line'):
            fine_sampled_report = psf_report(
                psf_type=psf_type,
                excitation_brightness=0.25,
                depletion_brightness=depletion_brightness,
                steps_per_excitation_psf_width=16,
                pulses_per_position=1,
                verbose=False)
            fine_resolution_improvement = fine_sampled_report[
                'resolution_improvement_descanned']
            steps_per_excitation_psf_width = 4 * fine_resolution_improvement
            report = psf_report(
                psf_type=psf_type,
                excitation_brightness=0.25,
                depletion_brightness=depletion_brightness,
                steps_per_excitation_psf_width=steps_per_excitation_psf_width,
                pulses_per_position=1,
                verbose=False)
            resolution_improvement[psf_type].append(
                report['resolution_improvement_descanned'])
            depletion_dose[psf_type].append(report['depletion_dose'])
    print('\nDone calculating.')
    fig = plt.figure(figsize=(12, 4), dpi=100)
    plt.plot(depletion_dose['point'], resolution_improvement['point'],
             '.', label='Descanned point STED')
    plt.plot(depletion_dose['line'], resolution_improvement['line'],
             '-', label='Descanned line STED')
    plt.xlim(-500, 12500)
    plt.ylim(1, 4.5)
    plt.legend(loc='lower right')
    plt.grid()
    plt.xlabel('Depletion dose (saturation units)')
    plt.ylabel('Resolution improvement (vs. diffraction limit)')
    plt.title('Resolution improvement vs. depletion dose\n' +
              'for variable step size and fixed excitation brightness')
    plt.savefig(os.path.join(output_directory,
                             '3_r_vs_depletion_dose_variable_steps.svg'),
                bbox_inches='tight')
    plt.close(fig)
    fig = plt.figure(figsize=(12, 4), dpi=100)
    plt.semilogx(depletion_dose['point'], resolution_improvement['point'],
             '.', label='Descanned point STED')
    plt.semilogx(depletion_dose['line'], resolution_improvement['line'],
             '-', label='Descanned line STED')
    plt.xlim(1e1, 2e4)
    plt.ylim(1, 4.5)
    plt.legend(loc='lower right')
    plt.grid()
    plt.xlabel('Depletion dose (saturation units)')
    plt.ylabel('Resolution improvement (vs. diffraction limit)')
    plt.title('Resolution improvement vs. depletion dose\n' +
              'for variable step size and fixed excitation brightness')
    plt.savefig(os.path.join(output_directory,
                             '4_r_vs_log_depletion_dose_variable_steps.svg'),
                bbox_inches='tight')
    plt.close(fig)

def r_vs_depletion_dose_variable_steps_variable_excitation():
    """A plot of resolution improvement vs. depletion dose, for a single
    line-STED orientation, but with a step size which varies to keep
    steps per STED PSF fixed, and an excitation brightness which varies
    to keep emitted signal constant:"""
    resolution_improvements = np.linspace(1, 4.6, 100)
    depletion_dose = {'descanned_point': [],
                      'descanned_line': [],
                      'rescanned_line': []}
    print("Calculating", end='')
    for r in resolution_improvements:
        print('.', sep='', end='')
        report = tune_psf(
            psf_type='point',
            scan_type='descanned',
            desired_resolution_improvement=r,
            desired_emissions_per_molecule=1.69,
            max_excitation_brightness=0.25,
            steps_per_improved_psf_width=4,
            relative_error=3e-2)
        depletion_dose['descanned_point'].append(report['depletion_dose'])
        report = tune_psf(
            psf_type='line',
            scan_type='descanned',
            desired_resolution_improvement=r,
            desired_emissions_per_molecule=1.69,
            max_excitation_brightness=0.25,
            steps_per_improved_psf_width=4,
            relative_error=3e-2)
        depletion_dose['descanned_line'].append(report['depletion_dose'])
        # This will fail for low resolution improvement, since rescan
        # confocal can't really do worse than root(2) resolution
        # improvement:
        if r > np.sqrt(2):
            report = tune_psf(
                psf_type='line',
                scan_type='rescanned',
                desired_resolution_improvement=r,
                desired_emissions_per_molecule=1.69,
                max_excitation_brightness=0.25,
                steps_per_improved_psf_width=4,
                relative_error=3e-2)
            depletion_dose['rescanned_line'].append(report['depletion_dose'])
        else:
            depletion_dose['rescanned_line'].append(0)
    print('\nDone calculating.')
    fig = plt.figure(figsize=(12, 4), dpi=100)
    plt.plot(depletion_dose['descanned_point'], resolution_improvements,
             '.', label='Descanned point STED')
    plt.plot(depletion_dose['descanned_line'], resolution_improvements,
             '-', label='Descanned line STED')
    plt.plot(depletion_dose['rescanned_line'], resolution_improvements,
             '--', label='Rescanned line STED')
    plt.ylim(1, 4.5)
    plt.legend(loc='lower right')
    plt.grid()
    plt.xlabel('Depletion dose (saturation units)')
    plt.ylabel('Resolution improvement (vs. diffraction limit)')
    plt.title('Resolution improvement vs. depletion dose\n' +
              'for variable step size and variable excitation brightness')
    plt.savefig(os.path.join(
        output_directory,
        '5_r_vs_depletion_dose_variable_steps_variable_exc.svg'),
                bbox_inches='tight')
    plt.close(fig)
    fig = plt.figure(figsize=(12, 4), dpi=100)
    plt.semilogx(depletion_dose['descanned_point'], resolution_improvements,
             '.', label='Descanned point STED')
    plt.semilogx(depletion_dose['descanned_line'], resolution_improvements,
             '-', label='Descanned line STED')
    plt.semilogx(depletion_dose['rescanned_line'], resolution_improvements,
             '--', label='Rescanned line STED')
    plt.xlim(1e1, 2e4)
    plt.ylim(1, 4.5)
    plt.legend(loc='lower right')
    plt.grid()
    plt.xlabel('Depletion dose (saturation units)')
    plt.ylabel('Resolution improvement (vs. diffraction limit)')
    plt.title('Resolution improvement vs. depletion dose\n' +
              'for variable step size and variable excitation brightness')
    plt.savefig(os.path.join(
        output_directory,
        '6_r_vs_log_depletion_dose_variable_steps_variable_exc.svg'),
                bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    main()
