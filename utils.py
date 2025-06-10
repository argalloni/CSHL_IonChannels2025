import matplotlib.pyplot as plt
import pyabf
import numpy as np
from scipy.optimize import curve_fit
from scipy.io import savemat, loadmat


###############################
# Data import functions
###############################

def get_sweeps(f):
    rec = pyabf.ABF(f)
    swps = []
    for swpNB in rec.sweepList:
        rec.setSweep(swpNB)
        swps.append((rec.sweepY,rec.sweepC))
    swps = np.array(swps)
    swp_time = rec.sweepX
    dt = swp_time[1] 
    return swps, swp_time, 1/dt


def abf_to_mat(f):
    swps, swp_time, sr = get_sweeps(f)
    mat = {'c001_Time\x00\x00\x00\xa0\x0f\x00\x00':swp_time}
    i = 2
    for swp in swps:
        mat[f'c{i:03}_Ipatch\x00\xa0\x0f\x00\x00'] = swp[0]
        i += 1
        mat[f'c{i:03}_Piezo_Com\x00\x00'] = swp[1]
        i += 1
        mat[f'c{i:03}_10Vm\x00\x00\x00\xa0\x0f\x00\x00'] = swp[1]
        i += 1
    savemat(f[:-4]+'.mat',mat)


def abf_to_mat_window(data_file, start_time, end_time):
    '''
    This function takes an .abf file and start/end times (in seconds) and returns a .mat file with the data in the window.
    '''
    sweeps, sweeps_time, sampling_freq = get_sweeps(data_file)
    I, time_in_window = select_sweep_window(sweeps, sweeps_time, start_time, end_time, sampling_freq, channel=0)
    V, time_in_window = select_sweep_window(sweeps, sweeps_time, start_time, end_time, sampling_freq, channel=1)
    mat = {'c001_Time\x00\x00\x00\xa0\x0f\x00\x00':sweeps_time}
    i = 2
    for swp in sweeps:
        mat[f'c{i:03}_Ipatch\x00\xa0\x0f\x00\x00'] = I
        i += 1
        mat[f'c{i:03}_Piezo_Com\x00\x00'] = V
        i += 1
        mat[f'c{i:03}_10Vm\x00\x00\x00\xa0\x0f\x00\x00'] = V
        i += 1
    savemat(f[:-4]+'.mat',mat)

###############################
# Data processing functions
###############################
def select_sweep_window(sweeps, time, start, end, sampling_freq, channel=0):
    if end != -1:
        assert start < end, "start_time must be before end_time"
    assert start < time[-1]*1000, f"start_time ({start} ms) must be before end of sweep: {time[-1]*1000:.3f} ms"

    i_start = int(start * sampling_freq / 1000)
    i_end = int(end * sampling_freq / 1000)
    return sweeps[:,channel,i_start:i_end], time[i_start:i_end]*1000


def get_step_measurements(sweeps, time, start_time, end_time, sampling_freq, measurement_type, abs=False):
    current_traces, time_i = select_sweep_window(sweeps, time, start_time, end_time, sampling_freq, channel=0)
    voltage_traces, time_v = select_sweep_window(sweeps, time, start_time, end_time, sampling_freq, channel=1)

    if measurement_type == 'mean':
        current_steps = np.mean(current_traces, axis=1)
        voltage_steps = np.mean(voltage_traces, axis=1)
    elif measurement_type == 'max':
        current_steps = np.max(current_traces, axis=1)
        voltage_steps = np.max(voltage_traces, axis=1)
    elif measurement_type == 'min':
        current_steps = np.min(current_traces, axis=1)
        voltage_steps = np.min(voltage_traces, axis=1)
    elif measurement_type == 'peak':
        current_traces_abs = np.abs(current_traces)
        current_steps_loc = np.argmax(current_traces_abs, axis=1)
        num_sweeps = len(current_traces)
        voltage_steps = voltage_traces[np.arange(num_sweeps), current_traces_abs.argmax(axis=1)]
        current_steps = current_traces[np.arange(num_sweeps), current_traces_abs.argmax(axis=1)]

    return voltage_steps, current_steps


def time_to_index(t, sampling_freq):
    """
    Convert time in milliseconds to index in the sweep, based on the sampling frequency.
    """
    return int(t * sampling_freq / 1000)


###############################
# Plotting functions
###############################
def update_plot_defaults():
    plt.rcParams.update({'font.size': 12,
                     'axes.spines.right': False,
                     'axes.spines.top': False,
                     'axes.linewidth':1.2,
                     'xtick.major.size': 6,
                     'xtick.major.width': 1.2,
                     'ytick.major.size': 6,
                     'ytick.major.width': 1.2,
                     'legend.frameon': False,
                     'legend.handletextpad': 0.1,
                     'svg.fonttype': 'none',
                     'text.usetex': False})


def plot_traces(time, voltage_traces, current_traces, marker_1=None, marker_2=None, ax=None):
    # Plot traces in the chosen window
    if ax is None:
        fig, ax = plt.subplots(2,1, figsize=(8, 8), sharex=False, height_ratios=(3, 1))
    ax[0].set_prop_cycle(color=plt.cm.viridis(np.linspace(0, 1, voltage_traces.shape[0])))
    ax[0].plot(time, voltage_traces.T, color='black', linewidth=0.5)
    ax[1].plot(time, current_traces.T, color='black', linewidth=0.8)
    ylims = ax[0].get_ylim()
    if marker_1 is not None:
        ax[0].vlines(marker_1, *ylims, color='red', linestyle='-', linewidth=0.5)
    if marker_2 is not None:
        ax[0].vlines(marker_2, *ylims, color='red', linestyle='-', linewidth=0.5)
    ax[1].set_xlabel('Time (ms)')
    ax[1].set_ylabel('Voltage (mV)')
    ax[0].set_ylabel('Current (pA)')
    ax[0].set_xlim(time[0], time[-1])
    ax[1].set_xlim(time[0], time[-1])
    return ax


def plot_IV(voltage, current, ax=None, xlabel_coords=None, ylabel_coords=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(voltage, current,'-o', markersize=4, color='black', linewidth=1)
    ax.set_xlabel('V (mV)')
    ax.set_ylabel('I (pA)')
    if np.max(current)>-100:
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if xlabel_coords is not None:
        ax.xaxis.set_label_coords(xlabel_coords[0], xlabel_coords[1])
    if ylabel_coords is not None:
        ax.yaxis.set_label_coords(ylabel_coords[0], ylabel_coords[1])

    # Customize ticks to remove the 0 ticks and labels
    xticks = [tick for tick in ax.get_xticks() if tick != 0]
    yticks = [tick for tick in ax.get_yticks() if tick != 0]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    return ax


def plot_expontial_fit(sweeps, sweeps_time, start_time, end_time, sampling_freq, ax=None):
    # Select the voltage only between the markers
    voltage_traces, t = select_sweep_window(sweeps, sweeps_time, start_time, end_time, sampling_freq, channel=0)

    def exp_decay(t, V0, tau, V_inf):
        return V0 * np.exp(-t / tau) + V_inf

    tau_values = []
    for trace in voltage_traces:
        # Initial guess: [V0, tau, V_inf]
        V0_guess = trace[0] - trace[-1]
        tau_guess = 50 / sampling_freq  # ms
        Vinf_guess = trace[-1]
        p0 = [V0_guess, tau_guess, Vinf_guess]

        popt, _ = curve_fit(exp_decay, t, trace, p0=p0, maxfev=2000)
        V0_fit, tau_fit, Vinf_fit = popt
        tau_values.append(tau_fit*1000)

        # Overlay fit
        fit_trace = exp_decay(t, *popt)
        if ax is not None:
            ax[0].plot(t*1000, fit_trace, color='r', alpha=1, linewidth=2)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(tau_values, 'o-')
    ax.set_xlabel('Sweep Index')
    ax.set_ylabel('Tau (ms)')
    ax.set_title('Tau Values for Each Sweep')

    return tau_values, ax