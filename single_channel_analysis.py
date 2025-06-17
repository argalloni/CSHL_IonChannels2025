from scipy import signal, stats
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


def detect_levels_from_histogram(traces, n_levels, plot_result=True, bins=200, mean_guesses=None, 
                                 removal_method='gaussian_subtraction', removal_factor=1.0):
    """
    Automatically detect levels in a single-channel current recording by fitting Gaussians the histogram of current values
    
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
    mean_guesses : list or None
        Optional list of initial guesses for the means of the Gaussians
        If provided, must have length equal to n_levels
    removal_method : str
        Method for removing fitted Gaussian influence:
        - 'gaussian_subtraction': subtract fitted Gaussian from histogram
        - 'data_masking': remove data points within N*std of fitted mean
        - 'weighted_subtraction': subtract with weights based on Gaussian probability
    removal_factor : float
        Factor for removal method (e.g., N*std for masking, or subtraction strength)
        
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
    
    # Store original histogram for plotting
    original_counts = counts.copy()
    
    # Data range for bounds
    data_min, data_max = np.min(all_currents), np.max(all_currents)
    data_range = data_max - data_min
    
    # Check if valid mean guesses were provided
    using_custom_means = False
    if mean_guesses is not None:
        if len(mean_guesses) == n_levels:
            using_custom_means = True
            mean_order = np.argsort(mean_guesses)  # Sort by value
            sorted_mean_guesses = [mean_guesses[i] for i in mean_order]
            print(f"Using provided mean guesses: {sorted_mean_guesses}")
        else:
            print(f"Warning: {len(mean_guesses)} mean guesses provided but {n_levels} levels requested.")
            print("Ignoring provided guesses and using automatic detection.")
    
    # Store results for each fitted Gaussian
    fitted_gaussians = []
    fitted_levels = []
    working_counts = counts.copy()
    working_data = all_currents.copy()
    
    fit_stats = {'amplitudes': [], 'means': [], 'stds': [], 'fit_success': True, 'r_squared_individual': []}
    
    # Iteratively fit Gaussians
    for level_idx in range(n_levels):
        print(f"\nFitting Gaussian {level_idx + 1}/{n_levels}...")
        
        # Determine initial guess for this level
        if using_custom_means:
            mean_guess = sorted_mean_guesses[level_idx]
            # Find amplitude at this mean
            closest_bin = np.argmin(np.abs(bin_centers - mean_guess))
            amp_guess = working_counts[closest_bin]
        else:
            # Find the highest peak in remaining histogram
            if np.max(working_counts) <= 0:
                print(f"No significant peaks remaining for level {level_idx + 1}")
                break
                
            peak_idx = np.argmax(working_counts)
            mean_guess = bin_centers[peak_idx]
            amp_guess = working_counts[peak_idx]
        
        # Conservative std guess
        std_guess = data_range / (n_levels * 4)
        
        # Fit single Gaussian to current working histogram
        try:
            # Parameter bounds for single Gaussian
            lower_bounds = [0, data_min - data_range*0.1, data_range/100]
            upper_bounds = [np.inf, data_max + data_range*0.1, data_range/2]
            
            popt, pcov = curve_fit(
                lambda x, a, m, s: a * np.exp(-0.5 * ((x - m) / s)**2),
                bin_centers, working_counts,
                p0=[amp_guess, mean_guess, std_guess],
                bounds=(lower_bounds, upper_bounds),
                maxfev=50000
            )
            
            amp_fit, mean_fit, std_fit = popt
            
            # Store fitted parameters
            fitted_gaussians.append(popt)
            fitted_levels.append(mean_fit)
            fit_stats['amplitudes'].append(amp_fit)
            fit_stats['means'].append(mean_fit)
            fit_stats['stds'].append(std_fit)
            
            # Calculate R-squared for this individual fit
            y_fitted = amp_fit * np.exp(-0.5 * ((bin_centers - mean_fit) / std_fit)**2)
            ss_res = np.sum((working_counts - y_fitted) ** 2)
            ss_tot = np.sum((working_counts - np.mean(working_counts)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            fit_stats['r_squared_individual'].append(r_squared)
            
            print(f"Fitted: mean={mean_fit:.3f} pA, std={std_fit:.3f} pA, R²={r_squared:.3f}")
            
            # Remove the influence of this Gaussian for next iteration
            if level_idx < n_levels - 1:  # Don't remove after last fit
                if removal_method == 'gaussian_subtraction':
                    # Subtract fitted Gaussian from histogram
                    fitted_gaussian = amp_fit * np.exp(-0.5 * ((bin_centers - mean_fit) / std_fit)**2)
                    working_counts = np.maximum(working_counts - removal_factor * fitted_gaussian, 
                                              np.zeros_like(working_counts))
                    
                elif removal_method == 'data_masking':
                    # Remove data points within N*std of fitted mean
                    mask = np.abs(working_data - mean_fit) > removal_factor * std_fit
                    working_data = working_data[mask]
                    if len(working_data) > 0:
                        working_counts, _ = np.histogram(working_data, bins=bin_centers, density=True)
                    else:
                        working_counts = np.zeros_like(working_counts)
                        
                elif removal_method == 'weighted_subtraction':
                    # Subtract with weights based on Gaussian probability
                    gaussian_weights = np.exp(-0.5 * ((bin_centers - mean_fit) / std_fit)**2)
                    fitted_gaussian = amp_fit * gaussian_weights
                    working_counts = np.maximum(working_counts - removal_factor * fitted_gaussian,
                                              0.1 * original_counts)  # Keep some minimum
                
        except Exception as e:
            print(f"  Fitting failed for level {level_idx + 1}: {e}")
            fit_stats['fit_success'] = False
            break
    
    # Sort levels by current value
    if fitted_levels:
        sort_order = np.argsort(fitted_levels)
        fitted_levels = [fitted_levels[i] for i in sort_order]
        
        # Reorder fit_stats accordingly
        for key in ['amplitudes', 'means', 'stds', 'r_squared_individual']:
            if key in fit_stats:
                fit_stats[key] = [fit_stats[key][i] for i in sort_order]
        
        # Calculate overall R-squared using all fitted Gaussians
        all_fitted = np.zeros_like(bin_centers)
        for i, (amp, mean, std) in enumerate(zip(fit_stats['amplitudes'], 
                                                fit_stats['means'], 
                                                fit_stats['stds'])):
            all_fitted += amp * np.exp(-0.5 * ((bin_centers - mean) / std)**2)
        
        ss_res = np.sum((original_counts - all_fitted) ** 2)
        ss_tot = np.sum((original_counts - np.mean(original_counts)) ** 2)
        fit_stats['r_squared'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
    else:
        print("No levels successfully fitted!")
        fitted_levels = []
        fit_stats['fit_success'] = False
    
    # Plot results (same as before, but using fitted parameters)
    if plot_result:
        import matplotlib.gridspec as gridspec
        
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # Create the subplots
        ax1 = fig.add_subplot(gs[0, 0])  # top left
        ax2 = fig.add_subplot(gs[0, 1])  # top right
        ax3 = fig.add_subplot(gs[1, :])  # bottom spanning both columns
        
        def plot_histogram_with_gaussians(ax, title_suffix="", ylim=None):
            """Helper function to create histogram plot with Gaussians"""
            ax.hist(all_currents, bins=bins, density=True, alpha=0.9, color='gray', edgecolor='white', label='Data')
            
            colors = plt.cm.Set1.colors[:len(fitted_levels)]
            if fit_stats.get('fit_success', False) and fitted_levels:
                # Plot individual Gaussians
                x_smooth = np.linspace(data_min, data_max, 1000)
                
                for i, (amp, mean, std) in enumerate(zip(fit_stats['amplitudes'], fit_stats['means'], fit_stats['stds'])):
                    individual_gaussian = amp * np.exp(-0.5 * ((x_smooth - mean) / std)**2)
                    ax.plot(x_smooth, individual_gaussian, '-', color=colors[i], 
                            label=f'Level {i+1}: {mean:.2f} pA', linewidth=3, alpha=0.8)
                    ax.axvline(mean, color=colors[i], linestyle='--', alpha=0.8, linewidth=2)
            else:
                # Just mark any detected levels
                for i, level in enumerate(fitted_levels):
                    ax.axvline(level, color=colors[i], linestyle='--', 
                            label=f'Level {i+1}: {level:.2f} pA')
            
            ax.set_xlabel('Current (pA)')
            ax.set_ylabel('Probability Density')
            ax.set_title(f'Current Histogram with {len(fitted_levels)} Fitted Levels{title_suffix}')
            ax.legend(handlelength=2, handletextpad=0.5, frameon=False)
            ax.grid(True, alpha=0.3)
            
            if fit_stats.get('fit_success', False):
                ax.text(0.02, 0.98, f"R² = {fit_stats['r_squared']:.3f}", 
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Set custom ylim if provided
            if ylim is not None:
                ax.set_ylim(ylim)
        
        # Plot 1: Full histogram (top left)
        plot_histogram_with_gaussians(ax1)
        
        # Plot 2: Zoomed histogram (top right) - half the y-axis range
        max_ylim = ax1.get_ylim()[1]
        plot_histogram_with_gaussians(ax2, title_suffix=" (Zoomed Y-axis)", ylim=(0, max_ylim / 20))
        
        # Plot 3: Sample traces with detected levels (bottom, spanning both columns)
        ax3.plot(traces.T, alpha=0.8, color='black', linewidth=0.5)
        
        colors_traces = plt.cm.Set1(np.linspace(0, 1, len(fitted_levels)))
        for i, level in enumerate(fitted_levels):
            ax3.axhline(level, color=colors_traces[i], linestyle='--', linewidth=2,
                    label=f'Level {i}: {level:.2f} pA')
        
        ax3.set_xlabel('Sample Number')
        ax3.set_ylabel('Current (pA)')
        ax3.set_title('Sample Traces with Detected Levels')
        ax3.legend(handlelength=2, handletextpad=0.5)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    # Print summary
    print(f"\n=== ITERATIVE LEVEL DETECTION RESULTS ===")
    print(f"Method: {removal_method} (factor: {removal_factor})")
    print(f"Detected {len(fitted_levels)} current levels:")
    for i, level in enumerate(fitted_levels):
        if i < len(fit_stats['r_squared_individual']):
            print(f"  Level {i}: {level:.3f} pA (individual R²: {fit_stats['r_squared_individual'][i]:.3f})")
        else:
            print(f"  Level {i}: {level:.3f} pA")
    
    if fit_stats.get('fit_success', False):
        print(f"Overall fit quality (R²): {fit_stats['r_squared']:.3f}")
        print("Standard deviations:", [f"{std:.3f}" for std in fit_stats['stds']])
    
    return fitted_levels, fit_stats


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
        self.hysteresis_factor = 0.05   # Fraction of level difference for hysteresis
        
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
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot 1: All traces overlaid
        time_ms = self.time_array * 1000
        ax = axes[0]
        ax.plot(time_ms, self.traces.T, 'b-', alpha=0.3, linewidth=0.5)
        ax.plot(time_ms, self.idealized_traces.T, 'r-', alpha=0.8, linewidth=1)
        ax.set_xlabel('Time')
        ax.set_ylabel('Current (pA)')
        ax.set_title('All Traces: Original (blue) vs Idealized (red)')
        ax.grid(True, alpha=0.3)
        
        # # Plot 2: Event count histogram
        # event_counts = [len(events) for events in self.event_lists if event[2]>0] # Count events other than baseline
        # axes[0,1].hist(event_counts, bins=max(1, max(event_counts)//2), alpha=0.7, edgecolor='black')
        # axes[0,1].set_xlabel('Number of Events per Trace')
        # axes[0,1].set_ylabel('Number of Traces')
        # axes[0,1].set_title(f'Event Count Distribution (Mean: {np.mean(event_counts):.1f})')
        # axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Event duration histogram
        # all_durations = []
        # for events in self.event_lists:
        #     for event in events:
        #         duration = (event[1] - event[0]) * 1000  # Convert to ms
        #         all_durations.append(duration)

        durations = self.calculate_event_durations(level_filter=1)
        all_durations = durations['L1']
        if all_durations:
            ax = axes[1]
            ax.hist(all_durations, bins=duration_hist_bins, alpha=0.7, edgecolor='white')
            ax.set_xlabel('Event Duration')
            ax.set_ylabel('Count')
            ax.set_title(f'Event Durations: Level 1 (n={len(all_durations)})')
            # ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        # # Plot 4: Level occupancy
        # level_counts = [0] * (len(self.current_levels) + 1)  # +1 for baseline
        # for events in self.event_lists:
        #     for event in events:
        #         level_idx = event[2]
        #         level_counts[level_idx] += 1
        # level_labels = ['Baseline'] + self.level_names
        # axes[1,1].bar(range(len(level_counts)), level_counts, alpha=0.7)
        # axes[1,1].set_xlabel('Level')
        # axes[1,1].set_ylabel('Number of Events')
        # axes[1,1].set_title('Level Occupancy')
        # axes[1,1].set_xticks(range(len(level_labels)))
        # axes[1,1].set_xticklabels(level_labels)
        # axes[1,1].grid(True, alpha=0.3)
        
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
            # All levels
            levels_to_analyze = range(0, len(self.current_levels) + 1)
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
    
    def plot_duration_histogram(self, level_filter=None, bins='auto', threshold=None, log_x=False, 
                            sqrt_y_scale=False, fit_gaussian=True, separate_plots=False):
        """
        Plot histogram of event durations with optional gaussian fitting

        Parameters:
        -----------
        level_filter : int, list, or None
            Which levels to include (None = all open levels)
        bins : int, str, or list
            Number of histogram bins or bin edges
        log_x : bool
            Use log scale for x-axis with logarithmically spaced bins
        sqrt_y_scale : bool
            Use square-root scale for y-axis
        fit_gaussian : bool
            Fit gaussian curve to durations
        separate_plots : bool
            Plot each level separately vs. overlaid
            
        Returns:
        --------
        durations : dict
            Dictionary with level names as keys, duration arrays as values
        threshold : float or None
            If fit_gaussian is True and level_filter includes baseline (0),
            returns the threshold duration where the two Gaussians cross
        """
        durations = self.calculate_event_durations(level_filter)

        if not any(durations.values()):
            print("No events found for specified levels")
            return durations, None

        if separate_plots and len(durations) > 1:
            n_levels = len(durations)
            fig, axes = plt.subplots(n_levels, 1, figsize=(6, 3.5 * n_levels))
            if n_levels == 1:
                axes = [axes]
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            axes = [ax]

        colors = plt.cm.Set2(np.linspace(0, 1, len(durations)))

        # Flatten all durations to get global min/max
        all_durations = np.concatenate([np.array(d) for d in durations.values() if len(d) > 0])
        all_durations = all_durations[all_durations > 0]  # to avoid issues with log scale

        # Global bin edges
        if log_x:
            global_min = np.min(all_durations)
            global_max = np.max(all_durations)
            bin_edges = np.logspace(np.log10(global_min), np.log10(global_max), 
                                bins if isinstance(bins, int) else 50)
        else:
            bin_edges = np.histogram_bin_edges(all_durations, bins=bins)

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
            level_durations = level_durations[level_durations > 0]

            counts, actual_bins, patches = current_ax.hist(
                level_durations, bins=bin_edges, alpha=0.7,
                color=colors[0], label=level_name,
                density=False, edgecolor='white', linewidth=0.5)

            if threshold is not None and level_name == 'Baseline':
                # Mark the threshold on the plot
                current_ax.axvline(threshold, color='r', linestyle='--', alpha=0.7,
                                label=f'Threshold: {threshold:.2f} ms')
                        
            # For baseline/closed durations, try fitting two Gaussians
            if fit_gaussian and level_name == 'Baseline' and len(level_durations) > 20:
                # First try a single Gaussian for reference
                mean_duration = np.mean(level_durations)
                std_duration = np.std(level_durations)
                
                # Convert bin edges to centers for plotting
                bin_centers = (actual_bins[:-1] + actual_bins[1:]) / 2
                
                # Set up initial parameters for two Gaussians
                # We'll assume one component for short closures and one for long closures
                try:
                    # Initial guess for the two Gaussians
                    p0 = [
                        0.7 * len(level_durations), np.log10(mean_duration/3), 0.3,  # First Gaussian: weight, mean, std
                        0.3 * len(level_durations), np.log10(mean_duration*3), 0.6   # Second Gaussian: weight, mean, std
                    ]
                    
                    # Define a two-Gaussian model
                    def two_gaussians(x, amp1, mean1, std1, amp2, mean2, std2):
                        # Use log-normal distribution if using log scale for better fit
                        if log_x:
                            x_log = np.log10(x)
                            g1 = amp1 * np.exp(-0.5 * ((x_log - mean1) / std1)**2) / (std1 * np.sqrt(2 * np.pi))
                            g2 = amp2 * np.exp(-0.5 * ((x_log - mean2) / std2)**2) / (std2 * np.sqrt(2 * np.pi))
                            return g1 + g2
                        else:
                            g1 = amp1 * np.exp(-0.5 * ((x - mean1) / std1)**2) / (std1 * np.sqrt(2 * np.pi))
                            g2 = amp2 * np.exp(-0.5 * ((x - mean2) / std2)**2) / (std2 * np.sqrt(2 * np.pi))
                            return g1 + g2
                    
                    # Define individual gaussian components for plotting
                    def gaussian_component(x, amp, mean, std):
                        if log_x:
                            x_log = np.log10(x)
                            return amp * np.exp(-0.5 * ((x_log - mean) / std)**2) / (std * np.sqrt(2 * np.pi))
                        else:
                            return amp * np.exp(-0.5 * ((x - mean) / std)**2) / (std * np.sqrt(2 * np.pi))
                    
                    # Fit the model to the histogram data
                    x_values = bin_centers
                    y_values = counts
                    
                    # Fit with bounds to constrain parameters to reasonable values
                    popt, _ = curve_fit(two_gaussians, x_values, y_values, p0=p0, 
                                    bounds=([0, -np.inf, 0.01, 0, -np.inf, 0.01], 
                                            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
                                    maxfev=10000)
                    
                    # Unpack the optimized parameters
                    amp1, mean1, std1, amp2, mean2, std2 = popt
                    
                    # Generate smooth curves for plotting
                    if log_x:
                        x_fit = np.logspace(np.log10(np.min(level_durations)), np.log10(np.max(level_durations)), 1000)
                    else:
                        x_fit = np.linspace(np.min(level_durations), np.max(level_durations), 1000)
                    
                    # Plot the individual components and combined fit
                    g1_fit = gaussian_component(x_fit, amp1, mean1, std1)
                    g2_fit = gaussian_component(x_fit, amp2, mean2, std2)
                    combined_fit = two_gaussians(x_fit, *popt)
                    
                    # Calculate real means (convert back from log if needed)
                    if log_x:
                        real_mean1 = 10**mean1
                        real_mean2 = 10**mean2
                    else:
                        real_mean1 = mean1
                        real_mean2 = mean2
                    
                    # Plot the components and combined fit
                    current_ax.plot(x_fit, g1_fit, 'g--', linewidth=1, 
                                label=f'Short closures (μ={real_mean1:.2f} ms)')
                    current_ax.plot(x_fit, g2_fit, 'b--', linewidth=1, 
                                label=f'Long closures (μ={real_mean2:.2f} ms)')
                    
                    # Find the threshold where the two Gaussians intersect (this is used to separate bursts)
                    intersection_mask = np.where(np.diff(np.sign(g1_fit - g2_fit)))[0]
                    if len(intersection_mask) > 0:
                        # Take the intersection point that's closest to the middle
                        middle_idx = len(x_fit) // 2
                        closest_idx = intersection_mask[np.argmin(np.abs(intersection_mask - middle_idx))]
                        threshold = x_fit[closest_idx]
                        print(f"Burst threshold identified at {threshold:.2f} ms")

                    if threshold is not None:
                        # Mark the threshold on the plot
                        current_ax.axvline(threshold, color='k', linestyle='--', alpha=0.7,
                                        label=f'Burst threshold: {threshold:.2f} ms')
                        
                    
                    print(f"Two-Gaussian fit for {level_name}:")
                    print(f"  Short closures: μ={real_mean1:.2f} ms, σ={std1:.2f}, weight={amp1:.2f}")
                    print(f"  Long closures: μ={real_mean2:.2f} ms, σ={std2:.2f}, weight={amp2:.2f}")
                    
                except (RuntimeError, ValueError) as e:
                    print(f"Failed to fit two Gaussians: {e}")
                    # Fall back to single Gaussian
                    fit_curve = stats.norm.pdf(actual_bins, mean_duration, std_duration)
                    current_ax.plot(actual_bins, fit_curve * np.diff(bin_edges)[0] * len(level_durations), 'r-', linewidth=2,
                                label=f'Gaussian fit (μ={mean_duration:.2f} ms, σ={std_duration:.2f} ms)')
            
            # For non-baseline levels, use the original single Gaussian fit
            elif fit_gaussian and len(level_durations) > 10:
                mean_duration = np.mean(level_durations)
                std_duration = np.std(level_durations)
                fit_curve = stats.norm.pdf(actual_bins, mean_duration, std_duration)
                current_ax.plot(actual_bins, fit_curve * np.diff(bin_edges)[0] * len(level_durations), 'r-', linewidth=2,
                            label=f'Gaussian fit (μ={mean_duration:.2f} ms, σ={std_duration:.2f} ms)')
                print(f"{level_name} - Mean duration: {mean_duration:.2f} ms, "
                    f"Std duration: {std_duration:.2f} ms, N events: {len(level_durations)}")

            if log_x:
                current_ax.set_xscale('log')

            if sqrt_y_scale:
                current_ax.set_yscale("function", functions=(np.sqrt, np.square))

            current_ax.set_xlabel('Duration')
            current_ax.set_ylabel('Count')
            current_ax.set_title(f'Event Duration Distribution{title_suffix}')
            current_ax.legend(handlelength=1, handletextpad=0.5)
            current_ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

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

    def analyze_bursts(self, closed_duration_threshold):
        """
        Analyze channel activity in bursts/blocks separated by long closed intervals
        
        Parameters:
        -----------
        closed_duration_threshold : float
            Threshold duration (in ms) for closed/baseline events
            Closed durations longer than this threshold define the boundaries between bursts
            
        Returns:
        --------
        burst_data : list of dicts
            List containing data for each burst, where each dict contains:
            - 'trace_idx': trace index
            - 'start_time': burst start time (s)
            - 'end_time': burst end time (s)
            - 'duration': burst duration (ms)
            - 'events': list of events in this burst
            - 'open_probability': Po within this burst
            - 'open_events': number of open events in burst
            - 'open_time': total open time in burst (ms)
        summary : dict
            Summary statistics about the bursts
        """
        if not self.event_lists:
            raise ValueError("No events detected. Run detect_all_events() first.")
        
        burst_data = []
        
        # Process each trace to identify bursts
        for trace_idx, events in enumerate(self.event_lists):
            if not events:
                continue
            
            current_burst_events = []
            burst_start_time = events[0][0]  # Start time of first event
            
            for i, event in enumerate(events):
                start_time, end_time, level_idx, amplitude = event
                
                # If this is a baseline/closed event longer than threshold
                if level_idx == 0 and (end_time - start_time) * 1000 > closed_duration_threshold:
                    # If we have events in the current burst, save it
                    if current_burst_events:
                        # Last event in the burst is the current long closed event
                        # We set its end time as the burst end
                        burst_end_time = start_time
                        
                        # Calculate burst statistics
                        burst_duration = (burst_end_time - burst_start_time) * 1000  # ms
                        
                        # Calculate open time and Po within burst
                        open_time = 0
                        open_events = 0
                        
                        for burst_event in current_burst_events:
                            b_start, b_end, b_level, b_amp = burst_event
                            if b_level > 0:  # Open state
                                open_time += (b_end - b_start) * 1000  # ms
                                open_events += 1
                        
                        burst_po = open_time / burst_duration if burst_duration > 0 else 0
                        
                        # Store burst data
                        burst_data.append({
                            'trace_idx': trace_idx,
                            'start_time': burst_start_time,
                            'end_time': burst_end_time,
                            'duration': burst_duration,
                            'events': current_burst_events,
                            'open_probability': burst_po,
                            'open_events': open_events,
                            'open_time': open_time
                        })
                        
                        # Start a new burst after this long closed event
                        current_burst_events = []
                        burst_start_time = end_time
                    else:
                        # No events in current burst, just move the start time
                        burst_start_time = end_time
                else:
                    # Add this event to the current burst
                    current_burst_events.append(event)
            
            # Handle final burst if there are any events
            if current_burst_events:
                burst_end_time = events[-1][1]  # End time of last event
                burst_duration = (burst_end_time - burst_start_time) * 1000  # ms
                
                # Calculate open time and Po
                open_time = 0
                open_events = 0
                for burst_event in current_burst_events:
                    b_start, b_end, b_level, b_amp = burst_event
                    if b_level > 0:  # Open state
                        open_time += (b_end - b_start) * 1000  # ms
                        open_events += 1
                
                burst_po = open_time / burst_duration if burst_duration > 0 else 0
                
                # Store burst data
                burst_data.append({
                    'trace_idx': trace_idx,
                    'start_time': burst_start_time,
                    'end_time': burst_end_time,
                    'duration': burst_duration,
                    'events': current_burst_events,
                    'open_probability': burst_po,
                    'open_events': open_events,
                    'open_time': open_time
                })
        
        # Calculate summary statistics
        if burst_data:
            n_bursts = len(burst_data)
            mean_duration = np.mean([burst['duration'] for burst in burst_data])
            mean_po = np.mean([burst['open_probability'] for burst in burst_data])
            # total_open_time = sum(burst['open_time'] for burst in burst_data)
            # total_duration = sum(burst['duration'] for burst in burst_data)
            # overall_po = total_open_time / total_duration if total_duration > 0 else 0
            
            summary = {
                'n_bursts': n_bursts,
                'mean_burst_duration_ms': mean_duration,
                'mean_burst_po': mean_po,
                # 'overall_po_in_bursts': overall_po,
                'threshold_ms': closed_duration_threshold
            }
        else:
            summary = {
                'n_bursts': 0,
                'threshold_ms': closed_duration_threshold
            }
        
        # Print summary
        print(f"Burst Analysis (threshold: {closed_duration_threshold} ms)")
        print(f"Number of bursts: {summary['n_bursts']}")
        if summary['n_bursts'] > 0:
            print(f"Mean burst duration: {summary['mean_burst_duration_ms']:.2f} ms")
            print(f"Mean Po within bursts: {summary['mean_burst_po']:.4f}")
            # print(f"Overall Po within bursts: {summary['overall_po_in_bursts']:.4f}")
        self.summary = summary

        return burst_data, summary

    def plot_burst_analysis(self, burst_data, max_bursts_to_plot=4):
        """
        Plot the results of burst analysis
        
        Parameters:
        -----------
        burst_data : list
            Output from analyze_bursts method
        max_bursts_to_plot : int
            Maximum number of individual bursts to plot (default: 4)
        """
        if not burst_data:
            print("No bursts found to plot")
            return
        
        # Plot Po distribution for bursts
        fig, ax = plt.subplots(figsize=(8, 3))
        
        po_values = [burst['open_probability'] for burst in burst_data]
        ax.hist(po_values, bins=30, alpha=0.7, edgecolor='white')
        ax.set_xlabel('Open Probability (Po)')
        ax.set_ylabel('Number of Bursts')
        ax.set_title(f'Distribution of Open Probability in Bursts (n={len(burst_data)})')
        ax.axvline(np.mean(po_values), color='r', linestyle='--', 
                label=f'Mean Po: {np.mean(po_values):.4f}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Plot example bursts in 2x2 subplots
        if max_bursts_to_plot > 0:
            # Sort bursts by duration (descending) and select a subset
            sorted_bursts = sorted(burst_data, key=lambda x: x['duration'], reverse=True)
            bursts_to_plot = sorted_bursts[:min(max_bursts_to_plot, len(sorted_bursts))]
            
            # Create 2x2 subplot figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 5))
            axes = axes.flatten()  # Flatten to make indexing easier
            
            for i, burst in enumerate(bursts_to_plot):
                trace_idx = burst['trace_idx']
                start_time = burst['start_time']
                end_time = burst['end_time']
                
                # Convert to indices in the trace array
                start_idx = np.searchsorted(self.time_array, start_time)
                end_idx = np.searchsorted(self.time_array, end_time)
                
                # Get the section of the trace for this burst
                time_section = self.time_array[start_idx:end_idx]
                trace_section = self.traces[trace_idx, start_idx:end_idx]
                idealized_section = self.idealized_traces[trace_idx, start_idx:end_idx]
                
                # Plot the burst
                ax = axes[i]
                ax.plot(time_section, trace_section, 'b-', alpha=0.4, label='Raw')
                ax.plot(time_section, idealized_section, 'r-', linewidth=1, alpha=0.8, label='Idealized')
                
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Current (pA)')
                ax.set_title(f'Example burst {i+1} - Duration: {burst["duration"]:.1f} ms, Po: {burst["open_probability"]:.4f}', fontsize=12)
                ax.grid(True, alpha=0.3)

            # Hide any unused subplots if fewer than 4 bursts
            for j in range(len(bursts_to_plot), 4):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            plt.show()

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
        po_overall = self.summary['mean_burst_po']
        print(f"\nOPEN PROBABILITY:")
        print(f"  Mean P(open): {po_overall:.4f} ({po_overall*100:.2f}%)")

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

    def generate_analysis_report_old(self):
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



