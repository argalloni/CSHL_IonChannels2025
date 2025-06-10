from scipy import signal, stats
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

##############################
# Lowpass filter
##############################

def baseline_correction(data, sampling_freq, method='polynomial', **kwargs):
    """
    Multiple methods for baseline correction in single-channel recordings.
    These preserve step-like channel events while removing slow baseline drift.
    
    Parameters:
    -----------
    data : array-like
        Input signal(s). Can be 1D or 2D array.
    sampling_freq : float
        Sampling frequency in Hz
    method : str
        Method to use: 'polynomial', 'median_subtraction', 'percentile', 'running_minimum'
    **kwargs : additional parameters for specific methods
    
    Returns:
    --------
    corrected_data : ndarray
        Baseline-corrected signal
    baseline : ndarray
        Estimated baseline (for visualization)
    """
    
    if data.ndim == 1:
        data = data[np.newaxis, :]  # Make 2D for consistent processing
        squeeze_output = True
    else:
        squeeze_output = False
    
    corrected_traces = np.zeros_like(data)
    baselines = np.zeros_like(data)
    
    for i in range(data.shape[0]):
        trace = data[i, :]
        
        if method == 'polynomial':
            # Fit and subtract polynomial trend
            degree = kwargs.get('degree', 3)
            x = np.arange(len(trace))
            coeffs = np.polyfit(x, trace, degree)
            baseline = np.polyval(coeffs, x)
            corrected = trace - baseline
            
        elif method == 'median_subtraction':
            # Use median filter to estimate baseline
            # Window should be much longer than channel events but shorter than drift
            window_ms = kwargs.get('window_ms', 1000)  # 1 second default
            window_samples = int(window_ms * sampling_freq / 1000)
            
            # Make window odd
            if window_samples % 2 == 0:
                window_samples += 1
            
            baseline = median_filter(trace, size=window_samples, mode='nearest')
            corrected = trace - baseline
            
        elif method == 'percentile':
            # Rolling percentile baseline estimation
            window_ms = kwargs.get('window_ms', 500)
            percentile = kwargs.get('percentile', 10)  # Use 10th percentile
            window_samples = int(window_ms * sampling_freq / 1000)
            
            baseline = rolling_percentile(trace, window_samples, percentile)
            corrected = trace - baseline
            
        elif method == 'running_minimum':
            # Running minimum with smoothing
            window_ms = kwargs.get('window_ms', 200)
            smooth_ms = kwargs.get('smooth_ms', 50)
            window_samples = int(window_ms * sampling_freq / 1000)
            smooth_samples = int(smooth_ms * sampling_freq / 1000)
            
            # Running minimum
            baseline = running_minimum(trace, window_samples)
            # Smooth the baseline
            baseline = uniform_filter1d(baseline, smooth_samples)
            corrected = trace - baseline
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        corrected_traces[i, :] = corrected
        baselines[i, :] = baseline
    
    if squeeze_output:
        return corrected_traces[0], baselines[0]
    else:
        return corrected_traces, baselines


def rolling_percentile(data, window, percentile):
    """Rolling percentile calculation"""
    result = np.zeros_like(data)
    half_window = window // 2
    
    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        result[i] = np.percentile(data[start:end], percentile)
    
    return result


def running_minimum(data, window):
    """Running minimum calculation"""
    result = np.zeros_like(data)
    half_window = window // 2
    
    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        result[i] = np.min(data[start:end])
    
    return result


def adaptive_baseline_correction(data, sampling_freq, closed_level_percentile=20):
    """
    Adaptive method: estimate closed level and subtract slowly-varying component
    Works well when you have clear closed and open levels
    """
    if data.ndim == 1:
        data = data[np.newaxis, :]
        squeeze_output = True
    else:
        squeeze_output = False
    
    corrected_traces = np.zeros_like(data)
    
    for i in range(data.shape[0]):
        trace = data[i, :]
        
        # Estimate closed level (assuming most time spent closed)
        closed_level = np.percentile(trace, closed_level_percentile)
        
        # Find points likely to be at closed level
        threshold = closed_level + 0.1 * np.std(trace)  # Small tolerance
        closed_points = trace <= threshold
        
        # If we have enough closed points, interpolate baseline
        if np.sum(closed_points) > len(trace) * 0.1:  # At least 10% closed points
            x = np.arange(len(trace))
            closed_x = x[closed_points]
            closed_y = trace[closed_points]
            
            # Interpolate baseline from closed points
            baseline = np.interp(x, closed_x, closed_y)
            
            # Smooth the baseline
            baseline = uniform_filter1d(baseline, int(0.1 * sampling_freq))  # 100ms smoothing
            
            corrected = trace - baseline
        else:
            # Fallback to polynomial detrending
            x = np.arange(len(trace))
            coeffs = np.polyfit(x, trace, 2)
            baseline = np.polyval(coeffs, x)
            corrected = trace - baseline
        
        corrected_traces[i, :] = corrected
    
    if squeeze_output:
        return corrected_traces[0]
    else:
        return corrected_traces


##############################
# Highpass filter
##############################

def lowpass_filter(data, cutoff_freq, sampling_freq, filter_order=4, filter_type='butterworth'):
    """
    Apply a lowpass filter to remove high-frequency noise from single-channel recordings.
    
    Parameters:
    -----------
    data : array-like
        Input signal(s) to filter. Can be 1D or 2D array.
        If 2D, filtering is applied along axis=1 (each row is a trace).
    cutoff_freq : float
        Low-pass cutoff frequency in Hz. Frequencies above this will be attenuated.
        For single-channel recordings, typically 1-5 kHz depending on channel kinetics.
    sampling_freq : float
        Sampling frequency of the data in Hz.
    filter_order : int, optional
        Order of the filter (default=4). Higher order = steeper rolloff.
    filter_type : str, optional
        Type of filter: 'butterworth' (default), 'bessel', or 'gaussian'
        - Butterworth: Flat passband, sharp rolloff
        - Bessel: Preserves pulse shape, gentler rolloff  
        - Gaussian: Very smooth, good for noisy data
    
    Returns:
    --------
    filtered_data : ndarray
        Filtered signal with same shape as input.
        
    Notes:
    ------
    - Uses zero-phase filtering (filtfilt) to avoid phase distortion
    - For BK channels: cutoff around 2-5 kHz typically preserves kinetics
    - For faster channels (e.g., AMPA): may need higher cutoff (5-10 kHz)
    - Rule of thumb: cutoff should be 3-5x higher than fastest channel kinetics
    """
    
    # Normalize cutoff frequency (0 to 1, where 1 is Nyquist frequency)
    nyquist_freq = sampling_freq / 2
    normalized_cutoff = cutoff_freq / nyquist_freq
    
    # Check if cutoff frequency is valid
    if normalized_cutoff >= 1:
        raise ValueError(f"Cutoff frequency ({cutoff_freq} Hz) must be less than "
                        f"Nyquist frequency ({nyquist_freq} Hz)")
    
    if normalized_cutoff <= 0:
        raise ValueError(f"Cutoff frequency must be positive, got {cutoff_freq} Hz")
    
    # Design the filter based on type
    if filter_type.lower() == 'butterworth':
        b, a = signal.butter(filter_order, normalized_cutoff, btype='low')
    elif filter_type.lower() == 'bessel':
        b, a = signal.bessel(filter_order, normalized_cutoff, btype='low')
    elif filter_type.lower() == 'gaussian':
        # For Gaussian filter, use different approach
        return gaussian_filter(data, cutoff_freq, sampling_freq)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Apply zero-phase filtering
    if data.ndim == 1:
        # Single trace
        filtered_data = signal.filtfilt(b, a, data)
    elif data.ndim == 2:
        # Multiple traces - filter each row
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i, :] = signal.filtfilt(b, a, data[i, :])
    else:
        raise ValueError("Data must be 1D or 2D array")
    
    return filtered_data


def gaussian_filter(data, cutoff_freq, sampling_freq):
    """
    Apply Gaussian lowpass filter (very smooth, good for noisy data)
    """
    from scipy.ndimage import gaussian_filter1d
    
    # Convert cutoff frequency to standard deviation in samples
    # For Gaussian filter: cutoff ≈ sampling_freq / (2π * sigma)
    sigma = sampling_freq / (2 * np.pi * cutoff_freq)
    
    if data.ndim == 1:
        filtered_data = gaussian_filter1d(data, sigma)
    else:
        filtered_data = gaussian_filter1d(data, sigma, axis=1)
    
    return filtered_data


def optimal_cutoff_suggestion(sampling_freq, channel_type='BK'):
    """
    Suggest optimal cutoff frequencies based on channel type and sampling rate
    """
    suggestions = {
        'BK': {'min': 1000, 'max': 5000, 'recommended': 2000},
        'SK': {'min': 1000, 'max': 3000, 'recommended': 1500},
        'Kv': {'min': 2000, 'max': 8000, 'recommended': 4000},
        'Nav': {'min': 5000, 'max': 15000, 'recommended': 8000},
        'AMPA': {'min': 5000, 'max': 20000, 'recommended': 10000},
        'NMDA': {'min': 1000, 'max': 5000, 'recommended': 2000},
        'GABA': {'min': 1000, 'max': 5000, 'recommended': 2000}
    }
    
    if channel_type in suggestions:
        rec = suggestions[channel_type]
        max_possible = sampling_freq / 3  # Conservative limit
        
        print(f"Suggested cutoff frequencies for {channel_type} channels:")
        print(f"  Minimum: {rec['min']} Hz")
        print(f"  Recommended: {min(rec['recommended'], max_possible)} Hz")
        print(f"  Maximum: {min(rec['max'], max_possible)} Hz")
        print(f"  (Limited by sampling rate: {max_possible:.0f} Hz)")
        
        return min(rec['recommended'], max_possible)
    else:
        print(f"Unknown channel type: {channel_type}")
        return sampling_freq / 5  # Conservative default


##############################
# Fit Gaussian to histogram 
##############################

def multi_gaussian(x, *params):
    """
    Multi-gaussian function for fitting
    params = [amp1, mean1, std1, amp2, mean2, std2, ...]
    """
    n_gaussians = len(params) // 3
    y = np.zeros_like(x)
    
    for i in range(n_gaussians):
        amp = params[i*3]
        mean = params[i*3 + 1] 
        std = params[i*3 + 2]
        y += amp * np.exp(-0.5 * ((x - mean) / std)**2)
    
    return y


def detect_levels_from_histogram(traces, n_levels, plot_result=True, bins=200):
    """
    Automatically detect current levels by fitting multiple Gaussians to current histogram
    
    Parameters:
    -----------
    traces : array-like (n_sweeps, n_timepoints)
        Filtered current traces
    n_levels : int
        Number of Gaussian peaks to fit (including baseline)
        E.g., n_levels=3 means baseline + 2 open levels
    plot_result : bool
        Whether to plot the histogram and fitted Gaussians
    bins : int
        Number of histogram bins
        
    Returns:
    --------
    detected_levels : list
        Sorted list of detected current levels (means of Gaussians)
    fit_params : dict
        Dictionary containing fit parameters and statistics
    """
    
    # Flatten all traces into single array
    all_currents = traces.flatten()
    
    # Create histogram
    counts, bin_edges = np.histogram(all_currents, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Initial parameter estimation
    # Use histogram peaks as initial guesses
    from scipy.signal import find_peaks
    
    # Find prominent peaks in histogram
    peaks, properties = find_peaks(counts, height=np.max(counts)*0.1, distance=bins//20)
    
    if len(peaks) < n_levels:
        print(f"Warning: Only found {len(peaks)} peaks, but requested {n_levels} levels")
        print("Using evenly spaced levels as backup")
        # Fallback: evenly spaced levels
        data_range = np.max(all_currents) - np.min(all_currents)
        detected_levels = [np.min(all_currents) + i * data_range / (n_levels-1) 
                          for i in range(n_levels)]
        return sorted(detected_levels), {}
    
    # Sort peaks by position and take the n_levels highest ones
    peak_heights = counts[peaks]
    peak_positions = bin_centers[peaks]
    
    # Sort by height and take top n_levels peaks
    sorted_indices = np.argsort(peak_heights)[::-1][:n_levels]
    selected_peaks = peaks[sorted_indices]
    selected_positions = peak_positions[sorted_indices]
    
    # Sort selected peaks by position (current value)
    position_order = np.argsort(selected_positions)
    selected_positions = selected_positions[position_order]
    selected_peaks = selected_peaks[position_order]
    
    # Initial parameter guesses: [amp1, mean1, std1, amp2, mean2, std2, ...]
    initial_params = []
    
    for i, peak_idx in enumerate(selected_peaks):
        amp_guess = counts[peak_idx]
        mean_guess = bin_centers[peak_idx]
        std_guess = (np.max(all_currents) - np.min(all_currents)) / (n_levels * 4)  # Conservative estimate
        
        initial_params.extend([amp_guess, mean_guess, std_guess])
    
    # Set parameter bounds
    # Amplitudes: positive, means: within data range, stds: reasonable range
    data_min, data_max = np.min(all_currents), np.max(all_currents)
    data_range = data_max - data_min
    
    lower_bounds = []
    upper_bounds = []
    
    for i in range(n_levels):
        lower_bounds.extend([0, data_min, data_range/100])  # amp > 0, mean in range, std > small
        upper_bounds.extend([np.inf, data_max, data_range/2])  # reasonable std upper limit
    
    try:
        # Fit multiple Gaussians
        popt, pcov = curve_fit(multi_gaussian, bin_centers, counts, 
                              p0=initial_params, 
                              bounds=(lower_bounds, upper_bounds),
                              maxfev=5000)
        
        # Extract fitted parameters
        fitted_levels = []
        fit_stats = {'amplitudes': [], 'means': [], 'stds': [], 'fit_success': True}
        
        for i in range(n_levels):
            amp = popt[i*3]
            mean = popt[i*3 + 1]
            std = popt[i*3 + 2]
            
            fitted_levels.append(mean)
            fit_stats['amplitudes'].append(amp)
            fit_stats['means'].append(mean)
            fit_stats['stds'].append(std)
        
        # Sort levels by current value
        fitted_levels = sorted(fitted_levels)
        
        # Calculate R-squared
        y_fitted = multi_gaussian(bin_centers, *popt)
        ss_res = np.sum((counts - y_fitted) ** 2)
        ss_tot = np.sum((counts - np.mean(counts)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        fit_stats['r_squared'] = r_squared
        fit_stats['fit_params'] = popt
        
    except Exception as e:
        print(f"Gaussian fitting failed: {e}")
        print("Using histogram peak positions as fallback")
        fitted_levels = sorted(selected_positions)
        fit_stats = {'fit_success': False, 'error': str(e)}
    
    # Plot results
    if plot_result:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Histogram with fitted Gaussians
        ax1.hist(all_currents, bins=bins, density=True, alpha=0.9, color='lightblue', 
                edgecolor='black', label='Data')
        
        if fit_stats.get('fit_success', False):
            # Plot fitted curve
            x_smooth = np.linspace(data_min, data_max, 1000)
            y_fitted_smooth = multi_gaussian(x_smooth, *popt)
            ax1.plot(x_smooth, y_fitted_smooth, 'r-', linewidth=2, label='Fitted Gaussians')
            
            # Plot individual Gaussians
            colors = plt.cm.Set1(np.linspace(0, 1, n_levels))
            for i in range(n_levels):
                amp = popt[i*3]
                mean = popt[i*3 + 1]
                std = popt[i*3 + 2]
                
                individual_gaussian = amp * np.exp(-0.5 * ((x_smooth - mean) / std)**2)
                ax1.plot(x_smooth, individual_gaussian, '--', color=colors[i], 
                        label=f'Level {i+1}: {mean:.2f} pA')
                ax1.axvline(mean, color=colors[i], linestyle=':', alpha=0.8)
        else:
            # Just mark the detected levels
            colors = plt.cm.Set1(np.linspace(0, 1, len(fitted_levels)))
            for i, level in enumerate(fitted_levels):
                ax1.axvline(level, color=colors[i], linestyle='--', 
                           label=f'Level {i+1}: {level:.2f} pA')
        
        ax1.set_xlabel('Current (pA)')
        ax1.set_ylabel('Probability Density')
        ax1.set_title(f'Current Histogram with {n_levels} Fitted Levels')
        ax1.legend(handlelength=2, handletextpad=0.5, frameon=False)
        ax1.grid(True, alpha=0.3)
        
        if fit_stats.get('fit_success', False):
            ax1.text(0.02, 0.98, f"R² = {fit_stats['r_squared']:.3f}", 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: Sample traces with detected levels
        ax2.plot(traces.T, alpha=0.8, color='black', linewidth=0.5)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(fitted_levels)))
        for i, level in enumerate(fitted_levels):
            ax2.axhline(level, color=colors[i], linestyle='--', linewidth=2,
                       label=f'Level {i+1}: {level:.2f} pA')
        
        ax2.set_xlabel('Sample Number')
        ax2.set_ylabel('Current (pA)')
        ax2.set_title('Sample Traces with Detected Levels')
        ax2.legend(handlelength=2, handletextpad=0.5)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Print summary
    print(f"\n=== LEVEL DETECTION RESULTS ===")
    print(f"Detected {len(fitted_levels)} current levels:")
    for i, level in enumerate(fitted_levels):
        print(f"  Level {i+1}: {level:.3f} pA")
    
    if fit_stats.get('fit_success', False):
        print(f"Fit quality (R²): {fit_stats['r_squared']:.3f}")
        print("Standard deviations:", [f"{std:.3f}" for std in fit_stats['stds']])
    
    return fitted_levels, fit_stats


##############################
# Event detection
##############################

class MultiLevelEventDetector:
    """
    Interactive event detection for single-channel recordings with multiple conductance levels.
    Allows manual specification of up to 5 different current levels above baseline.
    """
    
    def __init__(self, traces, time_array, sampling_freq):
        """
        Initialize the event detector
        
        Parameters:
        -----------
        traces : array-like (n_sweeps, n_timepoints)
            Filtered current traces
        time_array : array-like
            Time values for each sample
        sampling_freq : float
            Sampling frequency in Hz
        """
        self.traces = np.array(traces)
        self.time_array = np.array(time_array)
        self.sampling_freq = sampling_freq
        self.n_sweeps, self.n_timepoints = self.traces.shape
        
        # Initialize level parameters
        self.baseline_level = None
        self.current_levels = []  # List of current levels above baseline
        self.level_names = []     # Names for each level (e.g., 'L1', 'L2', etc.)
        self.detection_thresholds = []  # Midpoint thresholds between levels
        
        # Detection parameters
        self.min_event_duration = 1.0  # ms
        self.hysteresis_factor = 0.1   # Fraction of level difference for hysteresis
        
        # Results storage
        self.idealized_traces = None
        self.event_lists = []  # List of events for each sweep
        
    def estimate_baseline(self, method='percentile', percentile=10):
        """
        Estimate baseline level from the data
        
        Parameters:
        -----------
        method : str
            'percentile', 'mode', or 'minimum'
        percentile : float
            Percentile to use if method='percentile'
        """
        if method == 'percentile':
            self.baseline_level = np.percentile(self.traces, percentile)
        elif method == 'mode':
            # Use histogram mode
            hist, bins = np.histogram(self.traces.flatten(), bins=100)
            self.baseline_level = bins[np.argmax(hist)]
        elif method == 'minimum':
            self.baseline_level = np.min(self.traces)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Estimated baseline level: {self.baseline_level:.2f} pA")
        return self.baseline_level
    
    def set_current_levels(self, levels, level_names=None):
        """
        Manually set current levels above baseline
        
        Parameters:
        -----------
        levels : list
            List of current values (in pA) for each conductance level
            Should be in ascending order from baseline
        level_names : list, optional
            Names for each level (default: 'L1', 'L2', etc.)
        """
        if len(levels) > 5:
            raise ValueError("Maximum 5 levels supported")
        
        # Ensure levels are sorted
        levels = sorted(levels)
        self.current_levels = levels
        
        if level_names is None:
            self.level_names = [f'L{i+1}' for i in range(len(levels))]
        else:
            if len(level_names) != len(levels):
                raise ValueError("Number of level names must match number of levels")
            self.level_names = level_names
        
        # Calculate detection thresholds (midpoints between levels)
        all_levels = [self.baseline_level] + self.current_levels
        self.detection_thresholds = []
        
        for i in range(len(all_levels) - 1):
            threshold = (all_levels[i] + all_levels[i+1]) / 2
            self.detection_thresholds.append(threshold)
        
        print(f"Set {len(levels)} current levels:")
        for i, (level, name) in enumerate(zip(self.current_levels, self.level_names)):
            print(f"  {name}: {level:.2f} pA")
        
        print(f"Detection thresholds: {[f'{t:.2f}' for t in self.detection_thresholds]} pA")

    def detect_events_single_trace_old(self, trace_idx, plot_result=False):
        """
        Detect events in a single trace using the specified levels
        
        Parameters:
        -----------
        trace_idx : int
            Index of trace to analyze
        plot_result : bool
            Whether to plot the results
            
        Returns:
        --------
        events : list
            List of events, each containing: [start_time, end_time, level_idx, amplitude]
        idealized : array
            Idealized trace
        """
        trace = self.traces[trace_idx, :]
        idealized = np.full_like(trace, self.baseline_level)
        events = []
        
        # Convert minimum duration to samples
        min_samples = int(self.min_event_duration * self.sampling_freq / 1000)

        # State tracking
        current_level = 0  # 0 = baseline, 1+ = levels above baseline
        event_start = None
        
        # Create separate thresholds for up and down transitions with hysteresis
        all_levels = [self.baseline_level] + self.current_levels
        
        # Thresholds for upward transitions (opening)
        thresholds_up = []
        # Thresholds for downward transitions (closing)
        thresholds_down = []
        
        for i in range(len(all_levels) - 1):
            level_diff = all_levels[i+1] - all_levels[i]
            hysteresis = self.hysteresis_factor * level_diff
            
            # Midpoint between levels
            midpoint = (all_levels[i] + all_levels[i+1]) / 2
            
            # Apply hysteresis
            thresholds_up.append(midpoint + hysteresis/2)
            thresholds_down.append(midpoint - hysteresis/2)
        
        # State machine for event detection
        for i, current_value in enumerate(trace):
            new_level = current_level
            
            # Determine which level the current value belongs to
            if current_value <= thresholds_down[0]:
                # Below first threshold - definitely baseline
                new_level = 0
            else:
                # Check each level from highest to lowest
                found_level = False
                
                # First check if we're above the highest threshold
                if len(thresholds_up) > 0 and current_value >= thresholds_up[-1]:
                    new_level = len(self.current_levels)
                    found_level = True
                
                if not found_level:
                    # Check intermediate levels
                    for level_idx in range(len(self.current_levels) - 1, -1, -1):
                        if level_idx == 0:
                            # Transition from baseline to L1
                            if current_value >= thresholds_up[0]:
                                new_level = 1
                                found_level = True
                                break
                        else:
                            # Transition between levels
                            if (current_value >= thresholds_down[level_idx] and 
                                current_value < thresholds_up[level_idx]):
                                new_level = level_idx
                                found_level = True
                                break
                            elif current_value >= thresholds_up[level_idx]:
                                new_level = level_idx + 1
                                found_level = True
                                break
                
                if not found_level:
                    new_level = 0  # Default to baseline if no level found
            
            # Apply some smoothing - only change level if it's stable for a few samples
            # This helps reduce noise-induced false transitions
            stable_samples = max(3, int(0.1 * self.sampling_freq / 1000))  # At least 3 samples or 0.1ms
            
            if new_level != current_level:
                # Check if this level change is stable over next few samples
                if i + stable_samples < len(trace):
                    future_values = trace[i:i+stable_samples]
                    stable = True
                    
                    for val in future_values:
                        temp_level = 0
                        if val <= thresholds_down[0]:
                            temp_level = 0
                        else:
                            for level_idx in range(len(self.current_levels) - 1, -1, -1):
                                if level_idx == 0:
                                    if val >= thresholds_up[0]:
                                        temp_level = 1
                                        break
                                else:
                                    if (val >= thresholds_down[level_idx] and 
                                        val < thresholds_up[level_idx]):
                                        temp_level = level_idx
                                        break
                                    elif val >= thresholds_up[level_idx]:
                                        temp_level = level_idx + 1
                                        break
                        
                        if temp_level != new_level:
                            stable = False
                            break
                    
                    if not stable:
                        new_level = current_level  # Keep current level if not stable
            
            # Handle level changes
            if new_level != current_level:
                # End previous event if it was long enough
                if event_start is not None and (i - event_start) >= min_samples:
                    end_time = self.time_array[i-1]
                    level_idx = current_level
                    amplitude = self.current_levels[current_level-1] if current_level > 0 else self.baseline_level
                    
                    events.append([self.time_array[event_start], end_time, level_idx, amplitude])
                    
                    # Fill idealized trace for this event
                    idealized[event_start:i] = amplitude
                
                # Start new event if moving to a level above baseline
                if new_level > 0:
                    event_start = i
                else:
                    event_start = None
                
                current_level = new_level
        
        # Handle final event
        if event_start is not None and (len(trace) - event_start) >= min_samples:
            end_time = self.time_array[-1]
            level_idx = current_level
            amplitude = self.current_levels[current_level-1] if current_level > 0 else self.baseline_level
            
            events.append([self.time_array[event_start], end_time, level_idx, amplitude])
            
            # Fill idealized trace for this final event
            idealized[event_start:] = amplitude
        
        if plot_result:
            self.plot_single_trace_result(trace_idx, events, idealized)
        
        return events, idealized

    def detect_events_single_trace(self, trace_idx, plot_result=False):
        """
        Detect events in a single trace using the specified levels
        
        Parameters:
        -----------
        trace_idx : int
            Index of trace to analyze
        plot_result : bool
            Whether to plot the results
            
        Returns:
        --------
        events : list
            List of events, each containing: [start_time, end_time, level_idx, amplitude]
        idealized : array
            Idealized trace
        """
        trace = self.traces[trace_idx, :]
        idealized = np.full_like(trace, self.baseline_level)
        events = []
        
        # Convert minimum duration to samples
        min_samples = int(self.min_event_duration * self.sampling_freq / 1000)

        # State tracking
        current_level = 0  # 0 = baseline, 1+ = levels above baseline
        event_start = 0  # Start tracking from the beginning for baseline events too
        
        # Create separate thresholds for up and down transitions with hysteresis
        all_levels = [self.baseline_level] + self.current_levels
        
        # Thresholds for upward transitions (opening)
        thresholds_up = []
        # Thresholds for downward transitions (closing)
        thresholds_down = []
        
        for i in range(len(all_levels) - 1):
            level_diff = all_levels[i+1] - all_levels[i]
            hysteresis = self.hysteresis_factor * level_diff
            
            # Midpoint between levels
            midpoint = (all_levels[i] + all_levels[i+1]) / 2
            
            # Apply hysteresis
            thresholds_up.append(midpoint + hysteresis/2)
            thresholds_down.append(midpoint - hysteresis/2)
        
        # State machine for event detection
        for i, current_value in enumerate(trace):
            new_level = current_level
            
            # Determine which level the current value belongs to
            if current_value <= thresholds_down[0]:
                # Below first threshold - definitely baseline
                new_level = 0
            else:
                # Check each level from highest to lowest
                found_level = False
                
                # First check if we're above the highest threshold
                if len(thresholds_up) > 0 and current_value >= thresholds_up[-1]:
                    new_level = len(self.current_levels)
                    found_level = True
                
                if not found_level:
                    # Check intermediate levels
                    for level_idx in range(len(self.current_levels) - 1, -1, -1):
                        if level_idx == 0:
                            # Transition from baseline to L1
                            if current_value >= thresholds_up[0]:
                                new_level = 1
                                found_level = True
                                break
                        else:
                            # Transition between levels
                            if (current_value >= thresholds_down[level_idx] and 
                                current_value < thresholds_up[level_idx]):
                                new_level = level_idx
                                found_level = True
                                break
                            elif current_value >= thresholds_up[level_idx]:
                                new_level = level_idx + 1
                                found_level = True
                                break
                
                if not found_level:
                    new_level = 0  # Default to baseline if no level found
            
            # Apply some smoothing - only change level if it's stable for a few samples
            # This helps reduce noise-induced false transitions
            stable_samples = max(3, int(0.1 * self.sampling_freq / 1000))  # At least 3 samples or 0.1ms
            
            if new_level != current_level:
                # Check if this level change is stable over next few samples
                if i + stable_samples < len(trace):
                    future_values = trace[i:i+stable_samples]
                    stable = True
                    
                    for val in future_values:
                        temp_level = 0
                        if val <= thresholds_down[0]:
                            temp_level = 0
                        else:
                            for level_idx in range(len(self.current_levels) - 1, -1, -1):
                                if level_idx == 0:
                                    if val >= thresholds_up[0]:
                                        temp_level = 1
                                        break
                                else:
                                    if (val >= thresholds_down[level_idx] and 
                                        val < thresholds_up[level_idx]):
                                        temp_level = level_idx
                                        break
                                    elif val >= thresholds_up[level_idx]:
                                        temp_level = level_idx + 1
                                        break
                        
                        if temp_level != new_level:
                            stable = False
                            break
                    
                    if not stable:
                        new_level = current_level  # Keep current level if not stable
            
            # Handle level changes
            if new_level != current_level:
                # End previous event if it was long enough (for any level, including baseline)
                if (i - event_start) >= min_samples:
                    end_time = self.time_array[i-1]
                    level_idx = current_level
                    amplitude = self.current_levels[current_level-1] if current_level > 0 else self.baseline_level
                    
                    events.append([self.time_array[event_start], end_time, level_idx, amplitude])
                    
                    # Fill idealized trace for this event
                    idealized[event_start:i] = amplitude
                    
                    # Start new event
                    event_start = i
                else:
                    # Event was too short, so we extend the previous event
                    # But we'll still change the current level for future checks
                    current_level = new_level
                    continue
                
                current_level = new_level
        
        # Handle final event
        if (len(trace) - event_start) >= min_samples:
            end_time = self.time_array[-1]
            level_idx = current_level
            amplitude = self.current_levels[current_level-1] if current_level > 0 else self.baseline_level
            
            events.append([self.time_array[event_start], end_time, level_idx, amplitude])
            
            # Fill idealized trace for this final event
            idealized[event_start:] = amplitude
        
        if plot_result:
            self.plot_single_trace_result(trace_idx, events, idealized)
        
        return events, idealized

    def validate_levels(self, trace_idx=0):
        """
        Help validate that the current levels are appropriate for the data
        """
        trace = self.traces[trace_idx, :]
        
        print(f"Data range: {np.min(trace):.2f} to {np.max(trace):.2f} pA")
        print(f"Baseline level: {self.baseline_level:.2f} pA")
        print("\nCurrent levels:")
        all_levels = [self.baseline_level] + self.current_levels
        
        for i, level in enumerate(all_levels):
            if i == 0:
                print(f"  Baseline: {level:.2f} pA")
            else:
                print(f"  Level {i}: {level:.2f} pA")
        
        print("\nThresholds:")
        for i in range(len(all_levels) - 1):
            level_diff = all_levels[i+1] - all_levels[i]
            hysteresis = self.hysteresis_factor * level_diff
            midpoint = (all_levels[i] + all_levels[i+1]) / 2
            
            thresh_up = midpoint + hysteresis/2
            thresh_down = midpoint - hysteresis/2
            
            print(f"  {all_levels[i]:.1f} ↔ {all_levels[i+1]:.1f}: "
                f"up={thresh_up:.2f}, down={thresh_down:.2f} pA")
        
        # Check if levels make sense
        data_in_levels = 0
        for level in all_levels:
            nearby_points = np.sum(np.abs(trace - level) < 2.0)  # Within 2 pA
            data_in_levels += nearby_points
            print(f"  Points within 2 pA of {level:.1f} pA: {nearby_points} ({nearby_points/len(trace)*100:.1f}%)")
        
        print(f"\nTotal points accounted for: {data_in_levels/len(trace)*100:.1f}%")

    def detect_all_events(self, plot_summary=True, duration_hist_bins='auto'):
        """
        Detect events in all traces
        
        Parameters:
        -----------
        plot_summary : bool
            Whether to plot summary of results
        """
        self.event_lists = []
        self.idealized_traces = np.zeros_like(self.traces)
        
        print(f"Processing {self.n_sweeps} traces...")
        
        for i in range(self.n_sweeps):
            events, idealized = self.detect_events_single_trace(i)
            self.event_lists.append(events)
            self.idealized_traces[i, :] = idealized
        
        if plot_summary:
            self.plot_detection_summary(duration_hist_bins=duration_hist_bins)
        
        return self.event_lists, self.idealized_traces
    
    def plot_single_trace_result(self, trace_idx, events, idealized):
        """Plot results for a single trace"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        time_ms = self.time_array * 1000
        
        # Plot original and idealized traces
        ax.plot(time_ms, self.traces[trace_idx, :], 'b-', alpha=0.3, linewidth=0.5, label='Filtered')
        ax.plot(time_ms, idealized, 'r-', linewidth=0.5, label='Idealized')
        
        # Plot level lines
        ax.axhline(self.baseline_level, color='k', linestyle='--', alpha=0.5, label='Baseline')
        for i, (level, name) in enumerate(zip(self.current_levels, self.level_names)):
            ax.axhline(level, color=f'C{i+2}', linestyle='--', alpha=0.7, label=name)
        
        # Plot detection thresholds
        for i, thresh in enumerate(self.detection_thresholds):
            ax.axhline(thresh, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Current (pA)')
        ax.set_title(f'Event Detection - Trace {trace_idx+1} ({len(events)} events)')
        ax.legend(handlelength=2, handletextpad=0.5)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_detection_summary(self, duration_hist_bins='auto'):
        """Plot summary of detection results across all traces"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Plot 1: All traces overlaid
        time_ms = self.time_array * 1000
        axes[0,0].plot(time_ms, self.traces.T, 'b-', alpha=0.3, linewidth=0.5)
        axes[0,0].plot(time_ms, self.idealized_traces.T, 'r-', alpha=0.8, linewidth=1)
        axes[0,0].set_xlabel('Time (ms)')
        axes[0,0].set_ylabel('Current (pA)')
        axes[0,0].set_title('All Traces: Original (blue) vs Idealized (red)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Event count histogram
        event_counts = [len(events) for events in self.event_lists]
        axes[0,1].hist(event_counts, bins=max(1, max(event_counts)//2), alpha=0.7, edgecolor='black')
        axes[0,1].set_xlabel('Number of Events per Trace')
        axes[0,1].set_ylabel('Number of Traces')
        axes[0,1].set_title(f'Event Count Distribution (Mean: {np.mean(event_counts):.1f})')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Event duration histogram
        all_durations = []
        for events in self.event_lists:
            for event in events:
                duration = (event[1] - event[0]) * 1000  # Convert to ms
                all_durations.append(duration)
        
        if all_durations:
            axes[1,0].hist(all_durations, bins=duration_hist_bins, alpha=0.7, edgecolor='white')
            axes[1,0].set_xlabel('Event Duration (ms)')
            axes[1,0].set_ylabel('Count')
            axes[1,0].set_title(f'Event Duration Distribution (n={len(all_durations)})')
            axes[1,0].set_yscale('log')
            axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Level occupancy
        level_counts = [0] * (len(self.current_levels) + 1)  # +1 for baseline
        for events in self.event_lists:
            for event in events:
                level_idx = event[2]
                level_counts[level_idx] += 1
        
        level_labels = ['Baseline'] + self.level_names
        axes[1,1].bar(range(len(level_counts)), level_counts, alpha=0.7)
        axes[1,1].set_xlabel('Level')
        axes[1,1].set_ylabel('Number of Events')
        axes[1,1].set_title('Level Occupancy')
        axes[1,1].set_xticks(range(len(level_labels)))
        axes[1,1].set_xticklabels(level_labels)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def calculate_event_durations(self, level_filter=None):
        """
        Calculate event durations for all detected events
        
        Parameters:
        -----------
        level_filter : int or list, optional
            Only include events from specific level(s). None = all levels except baseline
            
        Returns:
        --------
        durations : dict
            Dictionary with level names as keys, duration arrays as values
        """
        if not self.event_lists:
            raise ValueError("No events detected. Run detect_all_events() first.")
        
        durations = {}
        
        # Initialize duration lists for each level
        if level_filter is None:
            # All levels except baseline (level 0)
            levels_to_analyze = range(1, len(self.current_levels) + 1)
        elif isinstance(level_filter, int):
            levels_to_analyze = [level_filter]
        else:
            levels_to_analyze = level_filter
        
        for level_idx in levels_to_analyze:
            level_name = self.level_names[level_idx - 1] if level_idx > 0 else 'Baseline'
            durations[level_name] = []
        
        # Collect durations from all traces
        for events in self.event_lists:
            for event in events:
                start_time, end_time, level_idx, amplitude = event
                
                if level_idx in levels_to_analyze:
                    duration_ms = (end_time - start_time) * 1000  # Convert to ms
                    level_name = self.level_names[level_idx - 1] if level_idx > 0 else 'Baseline'
                    durations[level_name].append(duration_ms)
        
        return durations
    
    def calculate_open_probability(self, method='time_based'):
        """
        Calculate open probability (Po) for the channels
        
        Parameters:
        -----------
        method : str
            'time_based': fraction of time spent in open states
            'event_based': fraction of events that are openings
            
        Returns:
        --------
        po_overall : float
            Overall open probability (all levels combined)
        po_by_level : dict
            Open probability for each individual level
        statistics : dict
            Additional statistics
        """
        if not self.event_lists or self.idealized_traces is None:
            raise ValueError("No events detected. Run detect_all_events() first.")
        
        if method == 'time_based':
            return self._calculate_po_time_based()
        elif method == 'event_based':
            return self._calculate_po_event_based()
        else:
            raise ValueError("Method must be 'time_based' or 'event_based'")
    
    def _calculate_po_time_based(self):
        """Calculate Po based on fraction of time spent open"""
        total_time = 0
        open_time_overall = 0
        open_time_by_level = {name: 0 for name in self.level_names}
        
        for trace_idx in range(self.n_sweeps):
            trace_duration = len(self.idealized_traces[trace_idx]) / self.sampling_freq
            total_time += trace_duration
            
            # Count time at each level
            for level_idx in range(1, len(self.current_levels) + 1):
                level_name = self.level_names[level_idx - 1]
                level_value = self.current_levels[level_idx - 1]
                
                # Count samples at this level
                samples_at_level = np.sum(self.idealized_traces[trace_idx] == level_value)
                time_at_level = samples_at_level / self.sampling_freq
                
                open_time_by_level[level_name] += time_at_level
                open_time_overall += time_at_level
        
        # Calculate probabilities
        po_overall = open_time_overall / total_time if total_time > 0 else 0
        po_by_level = {name: time / total_time for name, time in open_time_by_level.items()}
        
        statistics = {
            'total_recording_time_s': total_time,
            'total_open_time_s': open_time_overall,
            'total_closed_time_s': total_time - open_time_overall,
            'method': 'time_based'
        }
        
        return po_overall, po_by_level, statistics
    
    def _calculate_po_event_based(self):
        """Calculate Po based on event frequency (less common for single channels)"""
        total_events = 0
        open_events = 0
        events_by_level = {name: 0 for name in self.level_names}
        
        for events in self.event_lists:
            for event in events:
                start_time, end_time, level_idx, amplitude = event
                total_events += 1
                
                if level_idx > 0:  # Open state
                    open_events += 1
                    level_name = self.level_names[level_idx - 1]
                    events_by_level[level_name] += 1
        
        po_overall = open_events / total_events if total_events > 0 else 0
        po_by_level = {name: count / total_events for name, count in events_by_level.items()}
        
        statistics = {
            'total_events': total_events,
            'open_events': open_events,
            'closed_events': total_events - open_events,
            'method': 'event_based'
        }
        
        return po_overall, po_by_level, statistics
    
    def plot_duration_histogram(self, level_filter=None, bins='auto', log_scale=False, 
                               fit_exponential=True, separate_plots=False):
        """
        Plot histogram of event durations with optional exponential fitting
        
        Parameters:
        -----------
        level_filter : int, list, or None
            Which levels to include (None = all open levels)
        bins : int or 'auto'
            Number of histogram bins
        log_scale : bool
            Use log scale for y-axis
        fit_exponential : bool
            Fit exponential decay to durations
        separate_plots : bool
            Plot each level separately vs. overlaid
        """
        durations = self.calculate_event_durations(level_filter)
        
        if not any(durations.values()):
            print("No events found for specified levels")
            return
        
        if separate_plots and len(durations) > 1:
            n_levels = len(durations)
            fig, axes = plt.subplots(n_levels, 1, figsize=(10, 4*n_levels))
            if n_levels == 1:
                axes = [axes]
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            axes = [ax]
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(durations)))
        
        for i, (level_name, level_durations) in enumerate(durations.items()):
            if not level_durations:
                continue
                
            if separate_plots and len(durations) > 1:
                current_ax = axes[i]
                title_suffix = f" - {level_name}"
            else:
                current_ax = axes[0]
                title_suffix = ""
            
            level_durations = np.array(level_durations)
            
            # Plot histogram
            counts, bin_edges, patches = current_ax.hist(
                level_durations, bins=bins, alpha=0.7, 
                color=colors[i], label=level_name,
                density=True, edgecolor='white', linewidth=0.5)
            
            # Fit exponential if requested
            if fit_exponential and len(level_durations) > 10:
                # Fit single exponential: f(t) = (1/tau) * exp(-t/tau)
                mean_duration = np.mean(level_durations)
                tau = mean_duration  # Maximum likelihood estimate for exponential
                
                # Generate fit curve
                t_fit = np.linspace(0, np.max(level_durations), 1000)
                fit_curve = (1/tau) * np.exp(-t_fit/tau)
                
                current_ax.plot(t_fit, fit_curve, 'r-', linewidth=2, 
                              label=f'Exp fit (τ={tau:.2f} ms)')
                
                print(f"{level_name} - Mean duration: {mean_duration:.2f} ms, "
                      f"Tau: {tau:.2f} ms, N events: {len(level_durations)}")
            
            if log_scale:
                current_ax.set_yscale('log')
            
            current_ax.set_xlabel('Duration (ms)')
            current_ax.set_ylabel('Probability Density')
            current_ax.set_title(f'Event Duration Distribution{title_suffix}')
            current_ax.legend(handlelength=2, handletextpad=0.5)
            current_ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return durations
    
    def generate_analysis_report(self):
        """
        Generate a comprehensive analysis report
        """
        if not self.event_lists:
            print("No events detected. Run detect_all_events() first.")
            return
        
        print("="*60)
        print("SINGLE-CHANNEL ANALYSIS REPORT")
        print("="*60)
        
        # Basic statistics
        total_events = sum(len(events) for events in self.event_lists)
        print(f"\nBASIC STATISTICS:")
        print(f"  Number of traces: {self.n_sweeps}")
        print(f"  Total events detected: {total_events}")
        print(f"  Events per trace: {total_events/self.n_sweeps:.1f} ± {np.std([len(events) for events in self.event_lists]):.1f}")
        
        # Current levels
        print(f"\nCURRENT LEVELS:")
        print(f"  Baseline: {self.baseline_level:.2f} pA")
        for i, (level, name) in enumerate(zip(self.current_levels, self.level_names)):
            print(f"  {name}: {level:.2f} pA")
        
        # Open probability
        po_overall, po_by_level, po_stats = self.calculate_open_probability()
        print(f"\nOPEN PROBABILITY:")
        print(f"  Overall Po: {po_overall:.4f} ({po_overall*100:.2f}%)")
        for level_name, po in po_by_level.items():
            print(f"  {level_name} Po: {po:.4f} ({po*100:.2f}%)")
        
        print(f"\nRECORDING TIME:")
        print(f"  Total: {po_stats['total_recording_time_s']:.1f} s")
        print(f"  Open: {po_stats['total_open_time_s']:.1f} s")
        print(f"  Closed: {po_stats['total_closed_time_s']:.1f} s")
        
        # Duration statistics
        durations = self.calculate_event_durations()
        print(f"\nEVENT DURATIONS:")
        for level_name, level_durations in durations.items():
            if level_durations:
                mean_dur = np.mean(level_durations)
                median_dur = np.median(level_durations)
                std_dur = np.std(level_durations)
                print(f"  {level_name}: {mean_dur:.2f} ± {std_dur:.2f} ms (median: {median_dur:.2f} ms, n={len(level_durations)})")
        
        print("="*60)


