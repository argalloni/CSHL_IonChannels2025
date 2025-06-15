import matplotlib.pyplot as plt
import pyabf
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from scipy.ndimage import maximum_filter1d
from pathlib import Path
import h5py

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
    return swps, swp_time, rec.dataRate



class Trace():
    '''class for a time series data trace containing current and voltage recordings.

    Parameters
    ----------
    current_data: np.ndarray | list, default=[]
        The current_data to be analysed.
    sampling_interval: float, default=1
        The sampling interval of the data in seconds.
    current_unit: str, default=''
        The physical unit of the y-axis.
    filename: str, default=''
        The filename of the trace.
    voltage_data: np.ndarray | list, default=None
        The voltage data (if available).
    voltage_unit: str, default=''
        The physical unit of the voltage data.
    ttl_data: np.ndarray | list, default=None
        The TTL pulse data (if available).
    ttl_unit: str, default='V'
        The physical unit of the TTL data.
    concatenate_sweeps: bool, default=True
        Whether sweeps are concatenated (True) or stored separately (False).

    Attributes
    ----------
    events: np.ndarray
        Detected events as 2d array.
    concatenate_sweeps: bool
        Whether data contains concatenated sweeps or separate sweeps.
    '''
    def __init__(self, current_data: np.ndarray | list=None, sampling_interval: float=1, current_unit: str='', 
                 filename: str='', voltage_data: np.ndarray | list=None, voltage_unit: str='', 
                 ttl_data: np.ndarray | list=None, ttl_unit: str='V',
                 concatenate_sweeps: bool=False):
        self.current_data = current_data
        self.voltage_data = voltage_data
        self.ttl_data = ttl_data
        self.sampling = sampling_interval
        self.events = []
        self.current_unit = current_unit
        self.voltage_unit = voltage_unit
        self.ttl_unit = ttl_unit
        self.filename = filename
        self.concatenate_sweeps = concatenate_sweeps

    def __repr__(self):
        """Return a string representation of the Trace object."""
        lines = []
        lines.append(f"Trace('{self.filename}')")
        lines.append("=" * 40)
        
        # Basic info
        if self.current_data is not None:
            if self.concatenate_sweeps:
                lines.append(f"data points: {len(self.current_data):,}")
                lines.append(f"Duration: {self.total_time:.3f} s")
            else:
                lines.append(f"Number of sweeps: {self.current_data.shape[0]}")
                lines.append(f"Points per sweep: {self.current_data.shape[1]:,}")
                lines.append(f"Duration per sweep: {self.total_time:.3f} s")
            lines.append(f"Sampling rate: {self.sampling_rate:.0f} Hz")
            lines.append(f"Sampling interval: {self.sampling*1000:.3f} ms")
            
            # Show available channels
            channels = ["Current"]
            if self.voltage_data is not None:
                channels.append("Voltage")
            if self.ttl_data is not None:
                channels.append("TTL")
            lines.append(f"Available channels: {', '.join(channels)}")
        else:
            lines.append("No data loaded")
            
        return "\n".join(lines)

    @property
    def current_data(self):
        return self._current_data

    @current_data.setter
    def current_data(self, data):
        if data is not None:
            # ensure data is float64 to avoid issues with minmax_scale
            self._current_data = data.astype(np.float64)
        else:
            self._current_data = None

    @property
    def voltage_data(self):
        return self._voltage_data

    @voltage_data.setter
    def voltage_data(self, data):
        if data is not None:
            self._voltage_data = data.astype(np.float64)
        else:
            self._voltage_data = None

    @property
    def ttl_data(self):
        return self._ttl_data

    @ttl_data.setter
    def ttl_data(self, data):
        if data is not None:
            self._ttl_data = data.astype(np.float64)
        else:
            self._ttl_data = None

    @property
    def sampling(self):
        return self._sampling

    @sampling.setter
    def sampling(self, value):
        if value < 0:
            raise ValueError('Sampling interval must be positive')
        self._sampling = value

    @property
    def sampling_rate(self):
        return np.round(1/self.sampling)

    @property
    def time(self):
        ''' Returns time axis as numpy array '''
        if self.current_data is not None:
            if self.concatenate_sweeps:
                return np.arange(0, len(self.current_data)) * self.sampling
            else:
                # Return time axis for a single sweep
                return np.arange(0, self.current_data.shape[1]) * self.sampling
        return None

    @property
    def time_ms(self):
        return self.time * 1000

    @property
    def total_time(self):
        ''' Returns the total duration of the recording '''
        if self.current_data is not None:
            if self.concatenate_sweeps:
                return len(self.current_data) * self.sampling
            else:
                # Return duration of a single sweep
                return self.current_data.shape[1] * self.sampling
        return None

    @property
    def total_time_ms(self):
        return self.total_time * 1000

    @property
    def num_sweeps(self):
        ''' Returns the number of sweeps '''
        if self.current_data is not None and not self.concatenate_sweeps:
            return self.current_data.shape[0]
        return 1

    @classmethod
    def from_axon_file(cls, filename: str, channels: int | list=0, scaling: float | list=1.0, 
                    units: str | list=None, load_voltage: bool=True, load_ttl: bool=False,
                    concatenate_sweeps: bool=False, recording_mode: str="V clamp"):
        ''' Loads data from an AXON .abf file.

        Parameters
        ----------
        filename: string
            Path of a .abf file.
        channels: int or list, default=0
            The recording channel(s) to load. If int, loads single channel as current data.
            If list, channels are assigned as: [current, voltage, ttl] based on load_voltage and load_ttl flags.
        scaling: float or list, default=1.0
            Scaling factor(s) applied to the data. If list, must match number of channels.
        unit: str or list, default=None
            Data unit(s), to be set when using scaling factor. If list, must match number of channels.
        load_voltage: bool, default=False
            Whether to load voltage data from a second channel.
        load_ttl: bool, default=False
            Whether to load TTL data from a third channel.
        concatenate_sweeps: bool, default=True
            Whether to concatenate sweeps or keep them separate.
        recording_mode: str, default="V clamp"
            Recording mode: "V clamp" (voltage clamp) or "I clamp" (current clamp).
            In V clamp: sweepY=current, sweepC=voltage
            In I clamp: sweepY=voltage, sweepC=current

        Returns
        -------
        Trace
            An initialized Trace object.

        Raises
        ------
        Exception
            If the file is not a valid .abf file.
        IndexError
            When the selected channel does not exist in the file.
        ValueError
            When parameters don't match the number of channels or invalid recording_mode.
        '''
        if not Path(filename).suffix.lower() == '.abf':
            raise Exception('Incompatible file type. Method only loads .abf files.')

        # Validate recording_mode
        valid_modes = ["V clamp", "I clamp", "voltage clamp", "current clamp"]
        if recording_mode not in valid_modes:
            raise ValueError(f"Invalid recording_mode '{recording_mode}'. Must be one of: {valid_modes}")
        
        # Normalize recording mode
        is_current_clamp = recording_mode.lower() in ["i clamp", "current clamp"]

        import pyabf
        abf_file = pyabf.ABF(filename)
        
        # Handle different input types for channels
        if isinstance(channels, int):
            channels_to_load = [channels]
            if load_voltage:
                # Only add voltage channel if it exists
                if channels + 1 in abf_file.channelList:
                    voltage_channel = channels + 1
                    channels_to_load.append(voltage_channel)
                elif len(abf_file.channelList) > 1:
                    voltage_channel = abf_file.channelList[1]
                    channels_to_load.append(voltage_channel)
                # If no additional channels available, we'll try sweepC later
            if load_ttl:
                # Only add TTL channel if it exists
                if channels + 2 in abf_file.channelList:
                    ttl_channel = channels + 2
                    channels_to_load.append(ttl_channel)
                elif len(abf_file.channelList) > 2:
                    ttl_channel = abf_file.channelList[-1]
                    channels_to_load.append(ttl_channel)
        else:
            channels_to_load = channels

        # Validate channels exist
        for ch in channels_to_load:
            if ch not in abf_file.channelList:
                raise IndexError(f'Channel {ch} does not exist. Available channels: {abf_file.channelList}')

        # Handle scaling and unit parameters
        num_expected_channels = 1  # Always have current
        if load_voltage:
            num_expected_channels += 1
        if load_ttl:
            num_expected_channels += 1
        
        if isinstance(scaling, (int, float)):
            scaling = [scaling] * num_expected_channels
        elif len(scaling) != num_expected_channels:
            raise ValueError(f"Number of scaling factors ({len(scaling)}) must match expected number of channels ({num_expected_channels})")

        if units is None:
            # Get units for available channels, fill in defaults for missing ones
            units = []
            for i, ch in enumerate(channels_to_load):
                units.append(abf_file.adcUnits[ch])
            # Add default units for voltage/TTL if they'll be loaded from sweepC
            if load_voltage and len(units) < 2:
                units.append('mV' if not is_current_clamp else 'pA')  # voltage in I clamp, current in V clamp
            if load_ttl and len(units) < 3:
                units.append('V')
        elif isinstance(units, str):
            units = [units] * num_expected_channels
        elif len(units) != num_expected_channels:
            raise ValueError(f"Number of units ({len(units)}) must match expected number of channels ({num_expected_channels})")

        # Load data - handle multiple sweeps
        if concatenate_sweeps:
            # Original behavior - concatenate all sweeps
            if is_current_clamp:
                # In current clamp: sweepY=voltage, sweepC=current
                # But we want current_data to contain current, so we need to get it from sweepC or another channel
                current_data = None
                voltage_data = None
                
                # Try to get current data from specified channel first
                if len(channels_to_load) > 0:
                    try:
                        # Check if the primary channel actually contains current data
                        primary_data = abf_file.data[channels_to_load[0]] * scaling[0]
                        # In current clamp, if we're loading from the primary channel, it's likely voltage
                        # We'll assume the user knows what they're doing with explicit channel specification
                        current_data = primary_data
                    except (IndexError, KeyError):
                        pass
                
                # If no explicit current channel or load_voltage is True, try to get current from sweepC
                if current_data is None or load_voltage:
                    try:
                        # Collect current data (sweepC) from all sweeps
                        all_current_data = []
                        all_voltage_data = []
                        
                        for sweep_idx in range(abf_file.sweepCount):
                            abf_file.setSweep(sweep_idx)
                            
                            # In current clamp: sweepY=voltage, sweepC=current
                            voltage_sweep = abf_file.sweepY * (scaling[1] if load_voltage and len(scaling) > 1 else 1.0)
                            all_voltage_data.append(voltage_sweep)
                            
                            if abf_file.sweepC is not None:
                                current_sweep = abf_file.sweepC * scaling[0]
                                all_current_data.append(current_sweep)
                        
                        if all_current_data:
                            current_data = np.concatenate(all_current_data)
                            if load_voltage:
                                voltage_data = np.concatenate(all_voltage_data)
                                
                    except Exception as e:
                        print(f"Warning: Could not load current/voltage data in current clamp mode: {e}")
                        # Fallback to treating primary channel as current
                        current_data = abf_file.data[channels_to_load[0]] * scaling[0]
                
            else:
                # Voltage clamp mode - original behavior
                current_data = abf_file.data[channels_to_load[0]] * scaling[0]
                voltage_data = None
                
                # Try to load voltage data from specified channel first
                if load_voltage and len(channels_to_load) > 1:
                    try:
                        voltage_data = abf_file.data[channels_to_load[1]] * scaling[1]
                    except (IndexError, KeyError):
                        print(f"Warning: Could not load voltage from channel {channels_to_load[1]}, trying sweepC...")
                        voltage_data = None
                
                # Fallback: Check for voltage in sweepC if requested but not found
                if load_voltage and voltage_data is None:
                    try:
                        all_voltage_data = []
                        for sweep_idx in range(abf_file.sweepCount):
                            abf_file.setSweep(sweep_idx)
                            if abf_file.sweepC is not None:
                                sweep_scaling = scaling[1] if len(scaling) > 1 else 1.0
                                all_voltage_data.append(abf_file.sweepC * sweep_scaling)
                        
                        if all_voltage_data:
                            voltage_data = np.concatenate(all_voltage_data)
                            print("Successfully loaded voltage data from sweepC")
                    except Exception as e:
                        print(f"Warning: Could not load voltage from sweepC: {e}")
            
            # Load TTL data (same for both modes)
            ttl_data = None
            if load_ttl and len(channels_to_load) > 2:
                try:
                    ttl_data = abf_file.data[channels_to_load[2]] * scaling[2]
                except (IndexError, KeyError):
                    print(f"Warning: Could not load TTL from channel {channels_to_load[2]}")
                    
        else:
            # New behavior - keep sweeps separate
            abf_file.setSweep(0)  # Start with first sweep to get dimensions
            sweep_length = len(abf_file.sweepY)
            num_sweeps = abf_file.sweepCount
            
            # Initialize arrays for all sweeps
            current_data = np.zeros((num_sweeps, sweep_length))
            voltage_data = None
            ttl_data = None
            
            if load_voltage:
                voltage_data = np.zeros((num_sweeps, sweep_length))
            if load_ttl:
                ttl_data = np.zeros((num_sweeps, sweep_length))
            
            # Load each sweep
            for sweep_idx in range(num_sweeps):
                if is_current_clamp:
                    # In current clamp: sweepY=voltage, sweepC=current
                    abf_file.setSweep(sweep_idx, channel=channels_to_load[0])
                    
                    # Try to get current from sweepC first
                    current_loaded = False
                    if hasattr(abf_file, 'sweepC') and abf_file.sweepC is not None:
                        current_data[sweep_idx] = abf_file.sweepC * scaling[0]
                        current_loaded = True
                    
                    # If sweepC not available, use primary channel (might be incorrectly assigned)
                    if not current_loaded:
                        current_data[sweep_idx] = abf_file.sweepY * scaling[0]
                        if sweep_idx == 0:
                            print("Warning: Using primary channel for current in current clamp mode - data might be swapped")
                    
                    # Load voltage data
                    if load_voltage:
                        # In current clamp, voltage is typically in sweepY
                        voltage_data[sweep_idx] = abf_file.sweepY * (scaling[1] if len(scaling) > 1 else 1.0)
                        
                else:
                    # Voltage clamp mode - original behavior
                    abf_file.setSweep(sweep_idx, channel=channels_to_load[0])
                    current_data[sweep_idx] = abf_file.sweepY * scaling[0]
                    
                    # Load voltage data
                    if load_voltage:
                        voltage_loaded = False
                        
                        # Try loading from specified channel first
                        if len(channels_to_load) > 1:
                            try:
                                abf_file.setSweep(sweep_idx, channel=channels_to_load[1])
                                voltage_data[sweep_idx] = abf_file.sweepY * scaling[1]
                                voltage_loaded = True
                            except (IndexError, KeyError):
                                pass
                        
                        # Fallback to sweepC if channel approach failed
                        if not voltage_loaded:
                            try:
                                abf_file.setSweep(sweep_idx, channel=channels_to_load[0])  # Reset to current channel
                                if hasattr(abf_file, 'sweepC') and abf_file.sweepC is not None:
                                    sweep_scaling = scaling[1] if len(scaling) > 1 else 1.0
                                    voltage_data[sweep_idx] = abf_file.sweepC * sweep_scaling
                                    voltage_loaded = True
                            except Exception:
                                pass
                        
                        # If still no voltage data, warn user
                        if not voltage_loaded and sweep_idx == 0:  # Only warn once
                            print("Warning: Could not load voltage data from any source")
                            voltage_data = None
                            break
                            
                # Load TTL data (same for both modes)
                if load_ttl and len(channels_to_load) > 2:
                    try:
                        abf_file.setSweep(sweep_idx, channel=channels_to_load[2])
                        ttl_data[sweep_idx] = abf_file.sweepY * scaling[2]
                    except (IndexError, KeyError):
                        if sweep_idx == 0:  # Only warn once
                            print(f"Warning: Could not load TTL from channel {channels_to_load[2]}")
                            ttl_data = None
                            break
        
        # Set units based on recording mode
        if is_current_clamp:
            # In current clamp: current_data contains current, voltage_data contains voltage
            current_unit = units[0] if 'A' in units[0] or 'pA' in units[0] else 'pA'  # Default current unit
            voltage_unit = units[1] if len(units) > 1 and ('V' in units[1] or 'mV' in units[1]) else 'mV'  # Default voltage unit
        else:
            # In voltage clamp: current_data contains current, voltage_data contains voltage
            current_unit = units[0]
            voltage_unit = units[1] if len(units) > 1 else 'mV'
        
        ttl_unit = units[2] if len(units) > 2 else 'V'

        return cls(current_data=current_data, sampling_interval=1/abf_file.sampleRate, 
                current_unit=current_unit, filename=Path(filename).name,
                voltage_data=voltage_data, voltage_unit=voltage_unit,
                ttl_data=ttl_data, ttl_unit=ttl_unit,
                concatenate_sweeps=concatenate_sweeps)

    @classmethod
    def from_axon_file2(cls, filename: str, channels: int | list=0, scaling: float | list=1.0, 
                    units: str | list=None, load_voltage: bool=False, load_ttl: bool=False,
                    concatenate_sweeps: bool=False):
        ''' Loads data from an AXON .abf file.

        Parameters
        ----------
        filename: string
            Path of a .abf file.
        channels: int or list, default=0
            The recording channel(s) to load. If int, loads single channel as current data.
            If list, channels are assigned as: [current, voltage, ttl] based on load_voltage and load_ttl flags.
        scaling: float or list, default=1.0
            Scaling factor(s) applied to the data. If list, must match number of channels.
        unit: str or list, default=None
            Data unit(s), to be set when using scaling factor. If list, must match number of channels.
        load_voltage: bool, default=False
            Whether to load voltage data from a second channel.
        load_ttl: bool, default=False
            Whether to load TTL data from a third channel.
        concatenate_sweeps: bool, default=True
            Whether to concatenate sweeps or keep them separate.

        Returns
        -------
        Trace
            An initialized Trace object.

        Raises
        ------
        Exception
            If the file is not a valid .abf file.
        IndexError
            When the selected channel does not exist in the file.
        ValueError
            When parameters don't match the number of channels.
        '''
        if not Path(filename).suffix.lower() == '.abf':
            raise Exception('Incompatible file type. Method only loads .abf files.')

        import pyabf
        abf_file = pyabf.ABF(filename)
        
        # Handle different input types for channels
        if isinstance(channels, int):
            channels_to_load = [channels]
            if load_voltage:
                # Only add voltage channel if it exists
                if channels + 1 in abf_file.channelList:
                    voltage_channel = channels + 1
                    channels_to_load.append(voltage_channel)
                elif len(abf_file.channelList) > 1:
                    voltage_channel = abf_file.channelList[1]
                    channels_to_load.append(voltage_channel)
                # If no additional channels available, we'll try sweepC later
            if load_ttl:
                # Only add TTL channel if it exists
                if channels + 2 in abf_file.channelList:
                    ttl_channel = channels + 2
                    channels_to_load.append(ttl_channel)
                elif len(abf_file.channelList) > 2:
                    ttl_channel = abf_file.channelList[-1]
                    channels_to_load.append(ttl_channel)
        else:
            channels_to_load = channels

        # Validate channels exist
        for ch in channels_to_load:
            if ch not in abf_file.channelList:
                raise IndexError(f'Channel {ch} does not exist. Available channels: {abf_file.channelList}')

        # Handle scaling and unit parameters
        num_expected_channels = 1  # Always have current
        if load_voltage:
            num_expected_channels += 1
        if load_ttl:
            num_expected_channels += 1
        
        if isinstance(scaling, (int, float)):
            scaling = [scaling] * num_expected_channels
        elif len(scaling) != num_expected_channels:
            raise ValueError(f"Number of scaling factors ({len(scaling)}) must match expected number of channels ({num_expected_channels})")

        if units is None:
            # Get units for available channels, fill in defaults for missing ones
            units = []
            for i, ch in enumerate(channels_to_load):
                units.append(abf_file.adcUnits[ch])
            # Add default units for voltage/TTL if they'll be loaded from sweepC
            if load_voltage and len(units) < 2:
                units.append('mV')
            if load_ttl and len(units) < 3:
                units.append('V')
        elif isinstance(units, str):
            units = [units] * num_expected_channels
        elif len(units) != num_expected_channels:
            raise ValueError(f"Number of units ({len(units)}) must match expected number of channels ({num_expected_channels})")

        # Load data - handle multiple sweeps
        if concatenate_sweeps:
            # Original behavior - concatenate all sweeps
            current_data = abf_file.data[channels_to_load[0]] * scaling[0]
            voltage_data = None
            ttl_data = None
            
            # Try to load voltage data from specified channel first
            if load_voltage and len(channels_to_load) > 1:
                try:
                    voltage_data = abf_file.data[channels_to_load[1]] * scaling[1]
                except (IndexError, KeyError):
                    # If channel approach fails, try sweepC
                    print(f"Warning: Could not load voltage from channel {channels_to_load[1]}, trying sweepC...")
                    voltage_data = None
            
            # Fallback: Check for voltage in sweepC if requested but not found
            if load_voltage and voltage_data is None:
                try:
                    abf_file.setSweep(0)  # Set to first sweep to access sweepC
                    if hasattr(abf_file, 'sweepC') and abf_file.sweepC is not None:
                        # For concatenated sweeps, we need to collect sweepC from all sweeps
                        all_voltage_data = []
                        for sweep_idx in range(abf_file.sweepCount):
                            abf_file.setSweep(sweep_idx)
                            if abf_file.sweepC is not None:
                                sweep_scaling = scaling[1] if len(scaling) > 1 else 1.0
                                all_voltage_data.append(abf_file.sweepC * sweep_scaling)
                        
                        if all_voltage_data:
                            voltage_data = np.concatenate(all_voltage_data)
                            print("Successfully loaded voltage data from sweepC")
                except Exception as e:
                    print(f"Warning: Could not load voltage from sweepC: {e}")
            
            # Load TTL data
            if load_ttl and len(channels_to_load) > 2:
                try:
                    ttl_data = abf_file.data[channels_to_load[2]] * scaling[2]
                except (IndexError, KeyError):
                    print(f"Warning: Could not load TTL from channel {channels_to_load[2]}")
                    
        else:
            # New behavior - keep sweeps separate
            abf_file.setSweep(0)  # Start with first sweep to get dimensions
            sweep_length = len(abf_file.sweepY)
            num_sweeps = abf_file.sweepCount
            
            # Initialize arrays for all sweeps
            current_data = np.zeros((num_sweeps, sweep_length))
            voltage_data = None
            ttl_data = None
            
            if load_voltage:
                voltage_data = np.zeros((num_sweeps, sweep_length))
            if load_ttl:
                ttl_data = np.zeros((num_sweeps, sweep_length))
            
            # Load each sweep
            for sweep_idx in range(num_sweeps):
                # Load current data
                abf_file.setSweep(sweep_idx, channel=channels_to_load[0])
                current_data[sweep_idx] = abf_file.sweepY * scaling[0]
                
                # Load voltage data
                if load_voltage:
                    voltage_loaded = False
                    
                    # Try loading from specified channel first
                    if len(channels_to_load) > 1:
                        try:
                            abf_file.setSweep(sweep_idx, channel=channels_to_load[1])
                            voltage_data[sweep_idx] = abf_file.sweepY * scaling[1]
                            voltage_loaded = True
                        except (IndexError, KeyError):
                            pass
                    
                    # Fallback to sweepC if channel approach failed
                    if not voltage_loaded:
                        try:
                            abf_file.setSweep(sweep_idx, channel=channels_to_load[0])  # Reset to current channel
                            if hasattr(abf_file, 'sweepC') and abf_file.sweepC is not None:
                                sweep_scaling = scaling[1] if len(scaling) > 1 else 1.0
                                voltage_data[sweep_idx] = abf_file.sweepC * sweep_scaling
                                voltage_loaded = True
                        except Exception:
                            pass
                    
                    # If still no voltage data, warn user
                    if not voltage_loaded and sweep_idx == 0:  # Only warn once
                        print("Warning: Could not load voltage data from any source")
                        voltage_data = None
                        break
                        
                # Load TTL data
                if load_ttl and len(channels_to_load) > 2:
                    try:
                        abf_file.setSweep(sweep_idx, channel=channels_to_load[2])
                        ttl_data[sweep_idx] = abf_file.sweepY * scaling[2]
                    except (IndexError, KeyError):
                        if sweep_idx == 0:  # Only warn once
                            print(f"Warning: Could not load TTL from channel {channels_to_load[2]}")
                            ttl_data = None
                            break
        
        # Set units
        current_unit = units[0]
        voltage_unit = units[1] if len(units) > 1 else 'mV'
        ttl_unit = units[2] if len(units) > 2 else 'V'

        return cls(current_data=current_data, sampling_interval=1/abf_file.sampleRate, 
                current_unit=current_unit, filename=Path(filename).name,
                voltage_data=voltage_data, voltage_unit=voltage_unit,
                ttl_data=ttl_data, ttl_unit=ttl_unit,
                concatenate_sweeps=concatenate_sweeps)

    @classmethod
    def from_heka_file(cls, filename: str, rectype: str, group: int=0, load_series: list=[], exclude_series: list=[], 
                       exclude_sweeps: dict={}, scaling: float=1, unit: str=None, resample: bool=True,
                       concatenate_sweeps: bool=True):
        ''' Loads data from a HEKA .dat file. Name of the PGF sequence needs to be specified.
        
        Note: TTL support for HEKA files would require additional implementation based on 
        the specific HEKA file structure and how TTL data is stored.

        Parameters
        ----------
        filename: string
            Path of a .dat file.
        rectype: string
            Name of the PGF sequence in the file to be loaded.
        group: int, default=1
            HEKA group to load data from. Note that HEKA groups are numbered starting from 1, but Python idexes from zero. 
            Hence, group 1 in HEKA is group 0 in Python. 
        load_series: list, default=[]
            List of HEKA series to load. Uses zero-indexing, i.e. HEKA series 1 is 0 in the list.
        exclude_series: list, default=[].
            List of HEKA series to exclude.
        exclude_sweeps: dict, default={}.
            Dictionary with sweeps to exclude from analysis. E.g. {2 : [4, 5]} excludes sweeps 4 & 5 from series 2.
        scaling: float, default=1e12
            Scaling factor applied to the data. Defaults to 1e12 (i.e. pA)
        unit: str, default=''
            Data unit, to be set when using scaling factor.
        resample: boolean, default=rue
            Resample data in case of sampling rate mismatch.
        concatenate_sweeps: bool, default=True
            Whether to concatenate sweeps or keep them separate.

        Returns
        -------
        Trace
            An initialized Trace object.

        Raises
        ------
        Exception or ValueError
            If the file is not a valid .dat file.
        IndexError
            When the group index is out of range.
        ValueError
            When the sampling rates of different series mismatch and resampling is set to False.
        '''
        if not Path(filename).suffix.lower() == '.dat':
            raise Exception('Incompatible file type. Method only loads .dat files.')

        import FileImport.HekaReader as heka
        bundle = heka.Bundle(filename)

        if group < 0 or group > len(bundle.pul.children) - 1:
            raise IndexError('Group index out of range')

        bundle_series = dict()
        for i, SeriesRecord in enumerate(bundle.pul[group].children):
            bundle_series.update({i: SeriesRecord.Label})

        if load_series == []:
            series_list = [series_number for series_number, record_type in bundle_series.items() \
                    if record_type == rectype and series_number not in exclude_series]
        else:
            load_series = [x for x in load_series if x not in exclude_series]
            series_list = [series_number for series_number, record_type in bundle_series.items() \
                    if record_type == rectype and series_number in load_series]

        series_data = []
        series_resistances = []
        all_sweeps = []  # Store individual sweeps for non-concatenated mode
        
        for series in series_list:
            sweep_data = []
            for sweep in range(bundle.pul[group][series].NumberSweeps):
                if series not in exclude_sweeps:
                    try:
                        sweep_single = bundle.data[group, series, sweep, 0]
                        sweep_data.append(sweep_single)
                        if not concatenate_sweeps:
                            all_sweeps.append(sweep_single)
                    except IndexError as e:
                        pass
                else:
                    if sweep not in exclude_sweeps[int(series)]:
                        try:
                            sweep_single = bundle.data[group, series, sweep, 0]
                            sweep_data.append(sweep_single)
                            if not concatenate_sweeps:
                                all_sweeps.append(sweep_single)
                        except IndexError as e:
                            pass
            pgf_series_index = sum(len(bundle.pul[i].children) for i in range(group)) + series
            series_data.append((np.array(sweep_data), bundle.pgf[pgf_series_index].SampleInterval))
            series_resistances.append((1/bundle.pul[group][series][0][0].GSeries) * 1e-6)

        max_sampling_interval = max([el[1] for el in series_data])
        
        if concatenate_sweeps:
            data = np.array([], dtype=np.float64)
            for i, dat in enumerate(series_data):
                if dat[1] < max_sampling_interval:
                    if not resample:
                        raise ValueError(f'Sampling interval of series {i} is smaller than maximum sampling interval of all series')
                    step = int(max_sampling_interval / dat[1])
                    data = np.append(data, dat[0].flatten()[::step])
                else:
                    data = np.append(data, dat[0].flatten())
        else:
            # Handle non-concatenated sweeps
            processed_sweeps = []
            for sweep in all_sweeps:
                # Apply resampling if needed (simplified for individual sweeps)
                processed_sweeps.append(sweep)
            
            if processed_sweeps:
                # Ensure all sweeps have the same length
                min_length = min(len(sweep) for sweep in processed_sweeps)
                data = np.array([sweep[:min_length] for sweep in processed_sweeps])
            else:
                data = np.array([])
        
        data_unit = unit if unit is not None else bundle.pul[group][series_list[0]][0][0].YUnit

        cls.excluded_sweeps = exclude_sweeps
        cls.exlucded_series = exclude_series
        cls.Rseries = series_resistances

        return cls(current_data=data * scaling, sampling_interval=max_sampling_interval, 
                   current_unit=data_unit, filename=Path(filename).name, 
                   concatenate_sweeps=concatenate_sweeps)

    @classmethod
    def from_h5_file(cls, filename: str, tracename: str='data', scaling: float=1e12, 
                     sampling: float=2e-5, unit: str='pA', ttl_tracename: str=None,
                     voltage_tracename: str=None, concatenate_sweeps: bool=True):
        ''' Loads data from an hdf5 file. Name of the dataset needs to be specified.

        Parameters
        ----------
        filename: str
            Path of the .h5 file to load.
        tracename: str, default='data'
            Name of the dataset in the file to be loaded as current data.
        scaling: float, default=1e12
            Scaling factor applied to the current data. Defaults to 1e12 (i.e. pA)
        sampling: float, default=2e-5
            The sampling interval of the data in seconds. Defaults to 20 microseconds (i.e. 50kHz sampling rate).
        unit: string, default='pA'
            Data unit string after scaling. Used for display purposes.
        ttl_tracename: str, default=None
            Name of the TTL dataset in the file. If None, no TTL data is loaded.
        voltage_tracename: str, default=None
            Name of the voltage dataset in the file. If None, no voltage data is loaded.
        concatenate_sweeps: bool, default=True
            Whether to concatenate sweeps or keep them separate.

        Returns
        -------
        Trace
            An initialized Trace object.

        Raises  
        ------
        FileNotFoundError
            When the specified file or traces are not found.
        '''
        with h5py.File(filename, 'r') as f:
            # Load current data
            path = f.visit(lambda key : key if isinstance(f[key], h5py.Dataset) and key.split('/')[-1] == tracename else None)
            if path is None:
                raise FileNotFoundError(f'Current trace "{tracename}" not found in file')
            current_data = f[path][:] * scaling
            
            # Load voltage data if specified
            voltage_data = None
            if voltage_tracename is not None:
                voltage_path = f.visit(lambda key : key if isinstance(f[key], h5py.Dataset) and key.split('/')[-1] == voltage_tracename else None)
                if voltage_path is None:
                    raise FileNotFoundError(f'Voltage trace "{voltage_tracename}" not found in file')
                voltage_data = f[voltage_path][:]
            
            # Load TTL data if specified
            ttl_data = None
            if ttl_tracename is not None:
                ttl_path = f.visit(lambda key : key if isinstance(f[key], h5py.Dataset) and key.split('/')[-1] == ttl_tracename else None)
                if ttl_path is None:
                    raise FileNotFoundError(f'TTL trace "{ttl_tracename}" not found in file')
                ttl_data = f[ttl_path][:]

        return cls(current_data=current_data, sampling_interval=sampling, current_unit=unit, 
                  filename=Path(filename).name, voltage_data=voltage_data, ttl_data=ttl_data,
                  concatenate_sweeps=concatenate_sweeps)

    def crop(self, timepoint: float, window: float = None, time_units: str = 's', 
            timepoint_2: float = None, preserve_metadata: bool = True):
        """
        Crop the trace data around specified timepoint(s) and return a new Trace object.
        
        Parameters
        ----------
        timepoint : float
            The center timepoint for cropping (or start timepoint if timepoint_2 is provided).
        window : float, optional
            The window size around the timepoint. If timepoint_2 is provided, this parameter is ignored.
            For single timepoint: data is cropped from (timepoint - window/2) to (timepoint + window/2).
        time_units : str, default='s'
            Time units for timepoint, window, and timepoint_2. Options: 's' (seconds), 'ms' (milliseconds).
        timepoint_2 : float, optional
            Second timepoint. If provided, data is cropped between timepoint and timepoint_2.
            The window parameter is ignored when this is specified.
        preserve_metadata : bool, default=True
            Whether to preserve metadata (events, excluded_sweeps, etc.) in the new Trace object.
        
        Returns
        -------
        Trace
            A new Trace object containing the cropped data.
        
        Raises
        ------
        ValueError
            If no current data is available, if timepoints are out of bounds, if window is invalid,
            or if time units are not recognized.
        """
        if self.current_data is None:
            raise ValueError("No current data available for cropping")
        
        # Convert time units to seconds
        if time_units in ['s', 'seconds']:
            timepoint_s = timepoint
            window_s = window if window is not None else None
            timepoint_2_s = timepoint_2 if timepoint_2 is not None else None
        elif time_units in ['ms', 'milliseconds']:
            timepoint_s = timepoint / 1000.0
            window_s = window / 1000.0 if window is not None else None
            timepoint_2_s = timepoint_2 / 1000.0 if timepoint_2 is not None else None
        else:
            raise ValueError(f"Unknown time units: {time_units}. Use 's' for seconds or 'ms' for milliseconds.")
        
        # Determine start and end times
        if timepoint_2_s is not None:
            # Crop between two timepoints
            start_time = min(timepoint_s, timepoint_2_s)
            end_time = max(timepoint_s, timepoint_2_s)
        else:
            # Crop around single timepoint with window
            if window_s is None:
                raise ValueError("Either 'window' or 'timepoint_2' must be specified")
            if window_s <= 0:
                raise ValueError("Window size must be positive")
            
            start_time = timepoint_s
            end_time = timepoint_s + window_s
        
        # Validate time bounds
        if start_time < 0:
            raise ValueError(f"Start time ({start_time:.6f} s) is before data start (0 s)")
        if end_time > self.total_time:
            raise ValueError(f"End time ({end_time:.6f} s) exceeds data duration ({self.total_time:.6f} s)")
        if start_time >= end_time:
            raise ValueError("Start time must be less than end time")
        
        # Convert times to sample indices
        start_idx = int(start_time / self.sampling)
        end_idx = int(end_time / self.sampling)
        
        # Ensure we don't exceed array bounds
        if self.concatenate_sweeps:
            max_idx = len(self.current_data)
        else:
            max_idx = self.current_data.shape[1]
        
        start_idx = max(0, start_idx)
        end_idx = min(max_idx, end_idx)
        
        if start_idx >= end_idx:
            raise ValueError("Invalid time window results in empty data")
        
        # Crop the data
        if self.concatenate_sweeps:
            # Handle 1D data (concatenated sweeps)
            cropped_current = self.current_data[start_idx:end_idx]
            cropped_voltage = self.voltage_data[start_idx:end_idx] if self.voltage_data is not None else None
            cropped_ttl = self.ttl_data[start_idx:end_idx] if self.ttl_data is not None else None
        else:
            # Handle 2D data (separate sweeps)
            cropped_current = self.current_data[:, start_idx:end_idx]
            cropped_voltage = self.voltage_data[:, start_idx:end_idx] if self.voltage_data is not None else None
            cropped_ttl = self.ttl_data[:, start_idx:end_idx] if self.ttl_data is not None else None
        
        # Create new filename indicating the crop
        if timepoint_2_s is not None:
            crop_info = f"_crop_{timepoint:.3f}to{timepoint_2:.3f}{time_units}"
        else:
            crop_info = f"_crop_{timepoint:.3f}±{window/2:.3f}{time_units}"
        
        new_filename = self.filename.replace('.', crop_info + '.') if self.filename else f"cropped_trace{crop_info}.dat"
        
        # Create new Trace object
        cropped_trace = Trace(
            current_data=cropped_current,
            sampling_interval=self.sampling,
            current_unit=self.current_unit,
            filename=new_filename,
            voltage_data=cropped_voltage,
            voltage_unit=self.voltage_unit,
            ttl_data=cropped_ttl,
            ttl_unit=self.ttl_unit,
            concatenate_sweeps=self.concatenate_sweeps
        )
        
        # Preserve metadata if requested
        if preserve_metadata:
            # Copy events if they exist and adjust their timing
            if hasattr(self, 'events') and len(self.events) > 0:
                # Filter events that fall within the cropped time window
                # and adjust their timing relative to the new start
                original_events = np.array(self.events)
                if original_events.size > 0:
                    # Assuming events are stored as [time, ...] format
                    if original_events.ndim == 1:
                        event_times = original_events
                    else:
                        event_times = original_events[:, 0]  # Assume first column is time
                    
                    # Convert to seconds if needed
                    event_times_s = event_times * self.sampling  # Convert from sample indices to seconds
                    
                    # Find events within the cropped window
                    mask = (event_times_s >= start_time) & (event_times_s < end_time)
                    if np.any(mask):
                        cropped_events = original_events[mask] if original_events.ndim > 1 else original_events[mask]
                        # Adjust timing - subtract the start time and convert back to sample indices
                        if cropped_events.size > 0:
                            if cropped_events.ndim == 1:
                                cropped_events = cropped_events - start_idx
                            else:
                                cropped_events[:, 0] = cropped_events[:, 0] - start_idx
                            cropped_trace.events = cropped_events
            
            # Copy other metadata attributes if they exist
            for attr in ['excluded_sweeps', 'excluded_series', 'Rseries']:
                if hasattr(self, attr):
                    setattr(cropped_trace, attr, getattr(self, attr))
        
        return cropped_trace

    def get_step_events(self, threshold: float, channel: str = 'ttl', edge: str = 'rising', 
                        polarity: str = 'positive', time_units: str = 's', sweep: int = None):
        '''Extract step event times from any channel data.
        
        Parameters
        ----------
        threshold: float
            Threshold value for detecting step events.
        channel: str, default='ttl'
            Which channel to analyze ('current', 'voltage', or 'ttl').
        edge: str, default='rising'
            Type of edge to detect ('rising', 'falling', or 'both').
        polarity: str, default='positive'
            Step polarity to detect ('positive' for steps above threshold, 'negative' for steps below threshold).
        time_units: str, default='s'
            Units for returned event times ('s' or 'ms').
        sweep: int, optional
            For 2D data (separate sweeps), specify which sweep to analyze.
            If None and data is 2D, analyzes all sweeps and returns a list of arrays.
            
        Returns
        -------
        np.ndarray or list of np.ndarray
            Array of event times in specified units. For 2D data when sweep is None,
            returns a list with one array per sweep.
            
        Raises
        ------
        ValueError
            If no data is available for the specified channel or invalid parameters.
        '''
        # Validate parameters
        if channel not in ['current', 'voltage', 'ttl']:
            raise ValueError("Channel must be 'current', 'voltage', or 'ttl'")
            
        if edge not in ['rising', 'falling', 'both']:
            raise ValueError("Edge must be 'rising', 'falling', or 'both'")
        
        if polarity not in ['positive', 'negative']:
            raise ValueError("Polarity must be 'positive' or 'negative'")
        
        if time_units not in ['s', 'seconds', 'ms', 'milliseconds']:
            raise ValueError("time_units must be 's', 'seconds', 'ms', or 'milliseconds'")
        
        # Get the appropriate data
        if channel == 'current':
            if self.current_data is None:
                raise ValueError("No current data available")
            data = self.current_data
        elif channel == 'voltage':
            if self.voltage_data is None:
                raise ValueError("No voltage data available")
            data = self.voltage_data
        else:  # ttl
            if self.ttl_data is None:
                raise ValueError("No TTL data available")
            data = self.ttl_data
        
        def find_step_events_in_trace(trace_data):
            """Helper function to find step events in a 1D trace"""
            # Apply polarity logic to the threshold comparison
            if polarity == 'positive':
                above_threshold = trace_data > threshold
            else:  # negative polarity
                above_threshold = trace_data < threshold
            
            if edge == 'rising':
                crossings = np.where(np.diff(above_threshold.astype(int)) == 1)[0]
            elif edge == 'falling':
                crossings = np.where(np.diff(above_threshold.astype(int)) == -1)[0]
            else:  # both
                crossings = np.where(np.abs(np.diff(above_threshold.astype(int))) == 1)[0]
            
            # Convert to times
            event_times = crossings * self.sampling
            
            # Convert to requested units
            if time_units in ['ms', 'milliseconds']:
                event_times = event_times * 1000
            
            return event_times
        
        # Handle different data structures
        if data.ndim == 1:
            # 1D data (concatenated sweeps)
            return find_step_events_in_trace(data)
        else:
            # 2D data (separate sweeps)
            if sweep == 'all':
                sweep=None
            if sweep is None:
                # Analyze all sweeps
                event_times_list = []
                for i in range(data.shape[0]):
                    sweep_events = find_step_events_in_trace(data[i])
                    event_times_list.append(sweep_events)
                return event_times_list
            elif isinstance(sweep, (int, float)):
                # Analyze specific sweep
                if sweep >= data.shape[0]:
                    raise ValueError(f"Sweep index {sweep} exceeds number of sweeps ({data.shape[0]})")
                return find_step_events_in_trace(data[sweep])

    def get_event_times(self, threshold: float, polarity: str = 'positive', 
                        time_units: str = 's', channel: str = 'current', 
                        min_distance: int = None, prominence: float = None,
                        sweep: int = None):
        """
        Find event times based on peaks above or below a threshold.
        
        Parameters
        ----------
        threshold : float
            Threshold value for peak detection. Peaks above this value (positive polarity)
            or below this value (negative polarity) will be detected.
        polarity : str, default='positive'
            Peak polarity to detect. Options: 'positive' (above threshold), 'negative' (below threshold).
        time_units : str, default='s'
            Units for returned event times. Options: 's' (seconds), 'ms' (milliseconds).
        channel : str, default='current'
            Which channel to analyze. Options: 'current', 'voltage'.
        min_distance : int, optional
            Minimum number of samples between detected peaks. Helps avoid double-counting
            closely spaced peaks.
        prominence : float, optional
            Required prominence of peaks. Helps filter out small fluctuations.
        sweep : int, optional
            For 2D data (separate sweeps), specify which sweep to analyze. 
            If None and data is 2D, analyzes all sweeps and returns a list of arrays.
        
        Returns
        -------
        numpy.ndarray or list of numpy.ndarray
            Event times in specified units. For 2D data when sweep_idx is None,
            returns a list with one array per sweep.
        
        Raises
        ------
        ValueError
            If invalid parameters are provided or required data is not available.
        ImportError
            If scipy is not available for peak detection.
        """
        try:
            from scipy.signal import find_peaks
        except ImportError:
            raise ImportError("scipy is required for peak detection. Please install scipy.")
        
        # Validate parameters
        if polarity not in ['positive', 'negative']:
            raise ValueError("polarity must be 'positive' or 'negative'")
        
        if time_units not in ['s', 'seconds', 'ms', 'milliseconds']:
            raise ValueError("time_units must be 's', 'seconds', 'ms', or 'milliseconds'")
        
        if channel not in ['current', 'voltage']:
            raise ValueError("channel must be 'current' or 'voltage'")
        
        # Get the appropriate data
        if channel == 'current':
            if self.current_data is None:
                raise ValueError("No current data available")
            data = self.current_data
        else:  # voltage
            if self.voltage_data is None:
                raise ValueError("No voltage data available")
            data = self.voltage_data
        
        def find_events_in_trace(trace_data):
            """Helper function to find events in a 1D trace"""
            if polarity == 'positive':
                # Find peaks above threshold
                # For positive peaks, we look for peaks in the original data
                # and then filter by threshold
                peak_indices, properties = find_peaks(trace_data, 
                                                    distance=min_distance,
                                                    prominence=prominence)
                # Filter peaks that are above threshold
                above_threshold = trace_data[peak_indices] >= threshold
                event_indices = peak_indices[above_threshold]
                
            else:  # negative polarity
                # Find peaks below threshold
                # For negative peaks, we invert the data and find peaks,
                # then filter by threshold
                inverted_data = -trace_data
                peak_indices, properties = find_peaks(inverted_data,
                                                    distance=min_distance, 
                                                    prominence=prominence)
                # Filter peaks that are below threshold (in original data)
                below_threshold = trace_data[peak_indices] <= threshold
                event_indices = peak_indices[below_threshold]
            
            # Convert indices to times
            event_times = event_indices * self.sampling
            
            # Convert to requested units
            if time_units in ['ms', 'milliseconds']:
                event_times = event_times * 1000
            
            if len(event_times) == 0:
                print("WARNING: No events detected, double check your detection settings")
            return event_times
        
        # Handle different data structures
        if data.ndim == 1:
            # 1D data (concatenated sweeps)
            return find_events_in_trace(data)
        
        else:
            # 2D data (separate sweeps)
            if sweep in ['all', None]:
                # Analyze all sweeps
                event_times_list = []
                for i in range(data.shape[0]):
                    sweep_events = find_events_in_trace(data[i])
                    event_times_list.append(sweep_events)
                return event_times_list
            elif isinstance(sweep, (int, float)):
                # Analyze specific sweep
                if sweep >= data.shape[0]:
                    raise ValueError(f"Sweep index {sweep} exceeds number of sweeps ({data.shape[0]})")
                return find_events_in_trace(data[sweep])
           
    def get_measurements(self, start_time: float, end_time: float, measurement_type: str = 'mean', 
                            time_units: str = 's', sweep: int = None):
        """
        Extract measurements from current and voltage data within a specified time window.
        
        Parameters
        ----------
        start_time : float
            Start time of the measurement window.
        end_time : float
            End time of the measurement window.
        measurement_type : str, default='mean'
            Type of measurement to extract. Options: 'mean', 'max', 'min', 'peak'.
        time_units : str, default='s'
            Time units for start_time and end_time. Options: 's' (seconds), 'ms' (milliseconds).
        sweep : int, optional
            For 2D data, specify which sweep to measure. If None, measurements from all sweeps are returned.
        
        Returns
        -------
        tuple or list of tuples
            For 1D data or when sweep is specified: (current_measurement, voltage_measurement)
            For 2D data when sweep is None: list of tuples, one per sweep
            voltage_measurement is None if no voltage data is available.
        
        Raises
        ------
        ValueError
            If no current data is available, if time window is invalid, or if units are not recognized.
        """
        if self.current_data is None:
            raise ValueError("No current data available")
        
        # Convert time to seconds if needed
        if time_units in ['s','seconds']:
            start_time_s = start_time
            end_time_s = end_time
        elif time_units in ['ms', 'milliseconds']:
            start_time_s = start_time / 1000.0
            end_time_s = end_time / 1000.0
        else:
            raise ValueError(f"Unknown time units: {time_units}. Use 's' for seconds or 'ms' for milliseconds.")
        
        if start_time_s < 0 or end_time_s > self.total_time:
            raise ValueError("Time window exceeds data bounds")
        
        if start_time_s >= end_time_s:
            raise ValueError("Start time must be less than end time")
        
        # Convert time to sample indices
        start_idx = int(start_time_s / self.sampling)
        end_idx = int(end_time_s / self.sampling)
        
        def calculate_measurement(data_window, measurement_type):
            """Helper function to calculate measurement based on type"""
            if measurement_type == 'mean':
                return np.mean(data_window)
            elif measurement_type == 'max':
                return np.max(data_window)
            elif measurement_type == 'min':
                return np.min(data_window)
            elif measurement_type == 'peak':
                # Find peak (maximum absolute value)
                abs_data = np.abs(data_window)
                peak_idx = np.argmax(abs_data)
                return data_window[peak_idx]
            else:
                raise ValueError(f"Unknown measurement type: {measurement_type}")
        
        if self.current_data.ndim == 1:
            # Handle 1D data (concatenated sweeps)
            current_window = self.current_data[start_idx:end_idx]
            voltage_window = None
            if self.voltage_data is not None:
                voltage_window = self.voltage_data[start_idx:end_idx]
            
            current_measurement = calculate_measurement(current_window, measurement_type)
            voltage_measurement = calculate_measurement(voltage_window, measurement_type) if voltage_window is not None else None            
        else:
            if sweep == 'all':
                sweep = None
            # Handle 2D data (separate sweeps)
            if sweep is not None:
                # Measure specific sweep
                if sweep >= self.current_data.shape[0]:
                    raise ValueError(f"Sweep index {sweep} exceeds number of sweeps ({self.current_data.shape[0]})")
                
                current_window = self.current_data[sweep, start_idx:end_idx]
                voltage_window = None
                if self.voltage_data is not None:
                    voltage_window = self.voltage_data[sweep, start_idx:end_idx]
                
                current_measurement = calculate_measurement(current_window, measurement_type)
                voltage_measurement = calculate_measurement(voltage_window, measurement_type) if voltage_window is not None else None
                
                return current_measurement, voltage_measurement
            else:
                # Measure all sweeps
                # measurements = []
                current_measurement = []
                voltage_measurement = []
                for i in range(self.current_data.shape[0]):
                    current_window = self.current_data[i, start_idx:end_idx]
                    voltage_window = None
                    if self.voltage_data is not None:
                        voltage_window = self.voltage_data[i, start_idx:end_idx]
                    
                    sweep_current_measurement = calculate_measurement(current_window, measurement_type)
                    sweep_voltage_measurement = calculate_measurement(voltage_window, measurement_type) if voltage_window is not None else None
                    
                    current_measurement.append(sweep_current_measurement)
                    voltage_measurement.append(sweep_voltage_measurement)  
                current_measurement = np.array(current_measurement)
                voltage_measurement = np.array(voltage_measurement)         

        return current_measurement, voltage_measurement

    def subtract_baseline(self, start_time: float = 0, end_time: float = 1, time_units: str = 'ms', channel: str = 'current'):
        """
        Subtract baseline current and voltage from the data using measurements from a specified time window.
        
        Parameters
        ----------
        start_time : float, default=0
            Start time of the baseline measurement window.
        end_time : float, default=1
            End time of the baseline measurement window.
        time_units : str, default='ms'
            Time units for start_time and end_time. Options: 's' (seconds), 'ms' (milliseconds).
        channel : str, default='current'
            Which channel(s) to apply baseline correction to. Options: 'current', 'voltage', 'all'.
        
        Raises
        ------
        ValueError
            If no current data is available, if baseline measurement fails, or if invalid channel is specified.
        """
        if self.current_data is None:
            raise ValueError("No current data available for baseline subtraction")
        
        # Validate channel parameter
        valid_channels = ['current', 'voltage', 'all']
        if channel not in valid_channels:
            raise ValueError(f"Invalid channel '{channel}'. Must be one of: {valid_channels}")
        
        # Check if voltage channel is requested but not available
        if channel in ['voltage', 'all'] and self.voltage_data is None:
            if channel == 'voltage':
                raise ValueError("Voltage data not available for baseline correction")
            else:
                # For 'all', just warn and proceed with current only
                print("Warning: Voltage data not available, applying baseline correction to current only")
                channel = 'current'
        
        try:
            # Get baseline measurements using mean values
            baseline_current, baseline_voltage = self.get_measurements(
                start_time=start_time, 
                end_time=end_time, 
                measurement_type='mean', 
                time_units=time_units
            )
            print("BASELINE SUBTRACTED:")
            if self.current_data.ndim == 1:
                # Handle 1D data (concatenated sweeps)
                if channel in ['current', 'all']:
                    print(f"Baseline current: {baseline_current}") 
                    self.current_data = self.current_data - baseline_current
                
                if channel in ['voltage', 'all'] and self.voltage_data is not None and baseline_voltage is not None:
                    print(f"Baseline voltage: {baseline_voltage}") 
                    self.voltage_data = self.voltage_data - baseline_voltage
                    
            else:
                # Handle 2D data (separate sweeps)
                # baseline_current and baseline_voltage are arrays with one value per sweep
                if channel in ['current', 'all']:
                    print(f"Baseline currents: {baseline_current}") 
                    for i in range(self.current_data.shape[0]):
                        self.current_data[i] = self.current_data[i] - baseline_current[i]
                    
                if channel in ['voltage', 'all'] and self.voltage_data is not None and baseline_voltage is not None:
                    print(f"Baseline voltages: {baseline_voltage}") 
                    for i in range(self.voltage_data.shape[0]):
                        self.voltage_data[i] = self.voltage_data[i] - baseline_voltage[i]
                    
        except Exception as e:
            raise ValueError(f"Failed to subtract baseline: {str(e)}")

    def plot(self, plot_current=True, plot_voltage=False, plot_ttl=False, height_ratios=None, 
            marker_1=None, marker_2=None, time_units='s', sweep=None, plot_mean=False):
        ''' Plots the trace with optional markers
        
        Parameters
        ----------
        plot_current: bool, default=True
            Whether to plot current data (if available).
        plot_voltage: bool, default=False
            Whether to plot voltage data as well (if available).
        plot_ttl: bool, default=False
            Whether to plot TTL data as well (if available).
        height_ratios: tuple, default=None
            Height ratios for subplots. If None, defaults based on number of plots:
            - Single plot: N/A
            - Two plots: (3, 1) or (1, 1) if both are non-current
            - Three plots: (3, 1, 1) if current included, (1, 1, 1) if current not included
        marker_1: float, default=None
            Time position for first marker. If provided, adds a vertical line at this position.
        marker_2: float, default=None
            Time position for second marker. If provided, adds a vertical line at this position.
        time_units: str, default='s'
            Units for marker positions. Options: 's' (seconds), 'ms' (milliseconds).
        sweep: int, 'all', or None, default=None
            Which sweep to plot if data is not concatenated. If None, plots first sweep.
            If 'all', plots all sweeps superimposed with mean in red.
        plot_mean: bool, default=False
            Whether to plot the mean trace in red when sweep='all'.
        
        Returns
        -------
        matplotlib.axes or tuple of matplotlib.axes
            Single axis if only one channel plotted, tuple of axes if multiple plots.
            Order depends on which channels are plotted.
            
        Raises
        ------
        ValueError
            If no channels are selected for plotting or if required data is not available.
        '''
        # Check that at least one channel is selected for plotting
        if not any([plot_current, plot_voltage, plot_ttl]):
            raise ValueError("At least one channel must be selected for plotting")
        
        # Check data availability and what we can actually plot
        has_current = self.current_data is not None and plot_current
        has_voltage = self.voltage_data is not None and plot_voltage
        has_ttl = self.ttl_data is not None and plot_ttl
        
        # Validate that we have data for the requested channels
        channels_to_plot = []
        channel_data = []
        channel_labels = []
        channel_units = []
        
        if plot_current:
            if self.current_data is None:
                raise ValueError("Current data not available for plotting")
            channels_to_plot.append('current')
            channel_labels.append('Current')
            channel_units.append(self.current_unit)
        
        if plot_voltage:
            if self.voltage_data is None:
                raise ValueError("Voltage data not available for plotting")
            channels_to_plot.append('voltage')
            channel_labels.append('Voltage')
            channel_units.append(self.voltage_unit)
        
        if plot_ttl:
            if self.ttl_data is None:
                raise ValueError("TTL data not available for plotting")
            channels_to_plot.append('ttl')
            channel_labels.append('TTL')
            channel_units.append(self.ttl_unit)
        
        # Convert marker positions to seconds if needed
        markers = []
        for marker in [marker_1, marker_2]:
            if marker is not None:
                markers.append(marker)
            else:
                markers.append(None)
        
        marker_1_s, marker_2_s = markers
        
        # Determine what data to plot based on sweep parameter
        def get_data_for_channel(channel_name):
            if channel_name == 'current':
                data = self.current_data
            elif channel_name == 'voltage':
                data = self.voltage_data
            elif channel_name == 'ttl':
                data = self.ttl_data
            
            if self.concatenate_sweeps:
                return data
            else:
                # Handle separate sweeps
                if sweep is None:
                    return data[0]  # First sweep
                elif sweep == 'all':
                    return data  # All sweeps
                elif isinstance(sweep, int):
                    if sweep >= self.num_sweeps:
                        raise ValueError(f"Sweep {sweep} does not exist. Available sweeps: 0-{self.num_sweeps-1}")
                    return data[sweep]
                else:
                    raise ValueError("sweep must be None, 'all', or a valid sweep index")
        
        # Get time axis
        time_axis = self.time
        if time_units in ['ms', 'milliseconds']:
            time_axis = time_axis * 1000
        
        # Determine number of subplots and height ratios
        num_plots = len(channels_to_plot)
        
        if height_ratios is None:
            if num_plots == 1:
                height_ratios = None
            elif num_plots == 2:
                # Give current channel more space if it's included
                if has_current:
                    height_ratios = (3, 1)
                else:
                    height_ratios = (1, 1)
            else:  # num_plots == 3
                # Give current channel more space if it's included
                if has_current:
                    height_ratios = (3, 1, 1)
                else:
                    height_ratios = (1, 1, 1)
        
        # Create subplots
        if num_plots == 1:
            fig, ax = plt.subplots(figsize=(10, 4))
            axes = [ax]
        else:
            fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2 + 2*num_plots), 
                                    sharex=True, height_ratios=height_ratios)
            if not isinstance(axes, (list, np.ndarray)):
                axes = [axes]
        
        # Plot each requested channel
        for i, (channel_name, label, unit) in enumerate(zip(channels_to_plot, channel_labels, channel_units)):
            ax = axes[i]
            data_to_plot = get_data_for_channel(channel_name)
            
            # Plot the data
            if not self.concatenate_sweeps and sweep == 'all':
                # Plot all sweeps
                if plot_mean:
                    # Plot individual sweeps with transparency
                    for sweep_idx in range(self.num_sweeps):
                        ax.plot(time_axis, data_to_plot[sweep_idx], color='gray', alpha=0.3, linewidth=0.5)
                    # Plot mean in red
                    mean_data = np.mean(data_to_plot, axis=0)
                    ax.plot(time_axis, mean_data, color='red', linewidth=1, label='Mean')
                else:
                    # Plot all sweeps in black
                    for sweep_idx in range(self.num_sweeps):
                        ax.plot(time_axis, data_to_plot[sweep_idx], color='black', alpha=1, linewidth=0.5)
            else:
                # Plot single trace
                ax.plot(time_axis, data_to_plot, color='black', linewidth=0.5)
            
            # Set labels and formatting
            ax.set_ylabel(f'{label} ({unit})')
            ax.grid(True, alpha=0.3)
        
        # Set x-axis label on bottom subplot
        if time_units in ['ms', 'milliseconds']:
            axes[-1].set_xlabel('Time (ms)')
        else:
            axes[-1].set_xlabel('Time (s)')
        
        # Add markers to all subplots
        for marker, label in [(marker_1_s, 'marker 1'), (marker_2_s, 'marker 2')]:
            if marker is not None:
                for ax in axes:
                    ylims = ax.get_ylim()
                    ax.vlines(marker, *ylims, color='red', linestyle='-', linewidth=0.8)
                    ax.text(marker, ylims[1], label, fontsize=10, color='red', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Return appropriate axes
        if num_plots == 1:
            return axes[0]
        else:
            return tuple(axes)

    def remove_sweeps_with_spikes(self, threshold: float, polarity: str = 'positive',
                                channel: str = 'current', min_distance: int = None,
                                prominence: float = None, return_info: bool = False):
        """
        Remove sweeps that contain spikes above or below a specified threshold.
        
        This method only works with 2D data (separate sweeps). For concatenated data,
        use get_event_times() to identify spike locations manually.
        
        Parameters
        ----------
        threshold : float
            Threshold value for spike detection. Sweeps with spikes above this value 
            (positive polarity) or below this value (negative polarity) will be removed.
        polarity : str, default='positive'
            Spike polarity to detect. Options: 'positive' (above threshold), 'negative' (below threshold).
        channel : str, default='current'
            Which channel to analyze for spikes. Options: 'current', 'voltage'.
        min_distance : int, optional
            Minimum number of samples between detected spikes.
        prominence : float, optional
            Required prominence of spikes.
        return_info : bool, default=False
            If True, returns information about which sweeps were removed.
        
        Returns
        -------
        dict or None
            If return_info=True, returns a dictionary with:
            - 'removed_sweeps': list of removed sweep indices
            - 'kept_sweeps': list of kept sweep indices  
            - 'total_removed': number of sweeps removed
            - 'total_kept': number of sweeps kept
            If return_info=False, returns None.
        
        Raises
        ------
        ValueError
            If data is concatenated (1D), if no data is available, or invalid parameters.
        """
        # Check if data is 2D (separate sweeps)
        if self.concatenate_sweeps:
            raise ValueError("Cannot remove individual sweeps from concatenated data. "
                            "This method only works with separate sweeps (concatenate_sweeps=False).")
        
        # Validate parameters (reuse validation from get_event_times)
        if polarity not in ['positive', 'negative']:
            raise ValueError("polarity must be 'positive' or 'negative'")
        
        if channel not in ['current', 'voltage']:
            raise ValueError("channel must be 'current' or 'voltage'")
        
        # Get the appropriate data
        if channel == 'current':
            if self.current_data is None:
                raise ValueError("No current data available")
            data = self.current_data
        else:  # voltage
            if self.voltage_data is None:
                raise ValueError("No voltage data available")
            data = self.voltage_data
        
        if data.ndim == 1:
            raise ValueError("Data appears to be 1D (concatenated). This method requires 2D data (separate sweeps).")
        
        num_sweeps = data.shape[0]
        sweeps_to_remove = []
        sweeps_to_keep = []
        
        # Check each sweep for spikes
        for sweep_idx in range(num_sweeps):
            try:
                # Use get_event_times to detect spikes in this sweep
                spike_times = self.get_event_times(
                    threshold=threshold,
                    polarity=polarity,
                    channel=channel,
                    min_distance=min_distance,
                    prominence=prominence,
                    sweep_idx=sweep_idx
                )
                
                # If any spikes detected, mark sweep for removal
                if len(spike_times) > 0:
                    sweeps_to_remove.append(sweep_idx)
                else:
                    sweeps_to_keep.append(sweep_idx)
                    
            except Exception as e:
                # If spike detection fails for this sweep, keep it but warn
                print(f"Warning: Spike detection failed for sweep {sweep_idx}: {e}")
                sweeps_to_keep.append(sweep_idx)
        
        # Remove the identified sweeps
        if len(sweeps_to_remove) > 0:
            # Remove from current data
            self.current_data = np.delete(self.current_data, sweeps_to_remove, axis=0)
            
            # Remove from voltage data if present
            if self.voltage_data is not None:
                self.voltage_data = np.delete(self.voltage_data, sweeps_to_remove, axis=0)
            
            # Remove from TTL data if present
            if self.ttl_data is not None:
                self.ttl_data = np.delete(self.ttl_data, sweeps_to_remove, axis=0)
            
            print(f"Removed {len(sweeps_to_remove)} sweeps with spikes "
                f"(kept {len(sweeps_to_keep)} sweeps)")
        else:
            print("No sweeps contained spikes above threshold")
        
        # Return information if requested
        if return_info:
            return {
                'removed_sweeps': sweeps_to_remove,
                'kept_sweeps': sweeps_to_keep,
                'total_removed': len(sweeps_to_remove),
                'total_kept': len(sweeps_to_keep)
            }
        
        return None

    def detrend(self, detrend_type: str='linear', num_segments: int=0):
        ''' Detrend the data. 

        Parameters
        ----------
        detrend_type: str, default='linear'
            Type of detrending. Options: 'linear', 'constant'
        num_segments: int, default=0
            Number of segments for detrending. Increase in case of non-linear trends in the data.

        Returns
        -------
        Trace
            The detrended Trace object.
        '''
        if self.current_data is None:
            raise ValueError("No data to detrend")
        
        if self.concatenate_sweeps or self.current_data.ndim == 1:
            # Original behavior for concatenated data
            num_data = self.current_data.shape[0]
            breaks = np.arange(num_data/num_segments, num_data, num_data/num_segments, dtype=np.int64) if num_segments > 1 else 0
            detrended = signal.detrend(self.current_data, bp=breaks, type=detrend_type)
            
            # Also detrend voltage data if available
            voltage_detrended = None
            if self.voltage_data is not None:
                voltage_detrended = signal.detrend(self.voltage_data, bp=breaks, type=detrend_type)
        else:
            # Detrend each sweep individually
            detrended = np.zeros_like(self.current_data)
            for i in range(self.current_data.shape[0]):
                num_data = self.current_data.shape[1]
                breaks = np.arange(num_data/num_segments, num_data, num_data/num_segments, dtype=np.int64) if num_segments > 1 else 0
                detrended[i] = signal.detrend(self.current_data[i], bp=breaks, type=detrend_type)
            
            # Also detrend voltage data if available
            voltage_detrended = None
            if self.voltage_data is not None:
                voltage_detrended = np.zeros_like(self.voltage_data)
                for i in range(self.voltage_data.shape[0]):
                    voltage_detrended[i] = signal.detrend(self.voltage_data[i], bp=breaks, type=detrend_type)

        return Trace(detrended, self.sampling, current_unit=self.current_unit, filename=self.filename,
                    voltage_data=voltage_detrended, voltage_unit=self.voltage_unit, 
                    concatenate_sweeps=getattr(self, 'concatenate_sweeps', True))

    def filter(self, line_freq: float=None, width: float=None, highpass: float=None, lowpass: float=None, order: int=4,
            savgol: float=None, hann: int=None, apply_to_voltage: bool=True):
        ''' Filters trace with a combination of line frequency, high- and lowpass filters.
        If both lowpass and savgol arguments are passed, only the lowpass filter is applied. 

        Parameters
        ----------
        line_freq: float, default=None
            Line noise filter frequency (Hz). Line noise is removed by spectrum interpolation.
        width: float, default=None
            Width of the line noise filter (Hz).
        highpass: float, default=None
            Highpass cutoff frequency (Hz).
        lowpass: float, default=None
            Lowpass cutoff frequency (Hz). Set to None to turn filtering off.
        order: int, default=4
            Order of the filter.
        savgol: float, default=None
            The time window for Savitzky-Golay smoothing (ms).
        hann: int, default=None
            The length of the Hann window (samples).
        apply_to_voltage: bool, default=True
            Whether to apply the same filtering to voltage data (if available).
            
        Returns
        -------
        Trace
            A filtered Trace object.
        '''
        if self.current_data is None:
            raise ValueError("No data to filter")
        
        def apply_filters(data, original_data):
            filtered_data = data.copy()
            nyq = 0.5 * self.sampling_rate

            if line_freq:
                from scipy.fftpack import rfft, irfft, rfftfreq
                if filtered_data.ndim == 1:
                    fft = rfft(filtered_data)
                    xf = rfftfreq(filtered_data.shape[0], 1 / self.sampling_rate)
                    multiples = 6
                    for freq in np.arange(line_freq, (multiples * line_freq), line_freq):
                        fft[np.where(xf > freq - width/2)[0][0]:np.where(xf > freq + width/2)[0][0]] = 0
                    filtered_data = irfft(fft)
                else:
                    for i in range(filtered_data.shape[0]):
                        fft = rfft(filtered_data[i])
                        xf = rfftfreq(filtered_data.shape[1], 1 / self.sampling_rate)
                        multiples = 6
                        for freq in np.arange(line_freq, (multiples * line_freq), line_freq):
                            fft[np.where(xf > freq - width/2)[0][0]:np.where(xf > freq + width/2)[0][0]] = 0
                        filtered_data[i] = irfft(fft)
                    
            if highpass:
                sos = signal.butter(order, highpass / nyq, btype='high', output='sos')
                if filtered_data.ndim == 1:
                    filtered_data = signal.sosfilt(sos, filtered_data)
                else:
                    for i in range(filtered_data.shape[0]):
                        filtered_data[i] = signal.sosfilt(sos, filtered_data[i])
                        
            if lowpass:
                assert lowpass < nyq, "The lowpass cutoff frequency must be less than the Nyquist frequenct (sampling rate / 2)"
                if savgol:
                    print('Warning: Two lowpass filters selected, Savgol filter is ignored.')
                sos = signal.butter(order, lowpass / nyq, btype='low', analog=False, output='sos', fs=None)
                if filtered_data.ndim == 1:
                    filtered_data = signal.sosfiltfilt(sos, filtered_data)
                else:
                    for i in range(filtered_data.shape[0]):
                        filtered_data[i] = signal.sosfiltfilt(sos, filtered_data[i])
                        
            elif savgol:
                window_length = int(savgol / 1000 / self.sampling)
                if filtered_data.ndim == 1:
                    filtered_data = signal.savgol_filter(filtered_data, window_length, polyorder=order)
                else:
                    for i in range(filtered_data.shape[0]):
                        filtered_data[i] = signal.savgol_filter(filtered_data[i], window_length, polyorder=order)
                        
            elif hann:
                win = signal.windows.hann(hann)
                if filtered_data.ndim == 1:
                    filtered_data = signal.convolve(filtered_data, win, mode='same') / sum(win)
                    # Hann window generates edge artifacts due to zero-padding. Retain unfiltered data at edges.
                    filtered_data[:hann] = original_data[:hann]
                    filtered_data[filtered_data.shape[0] - hann:] = original_data[filtered_data.shape[0] - hann:]
                else:
                    for i in range(filtered_data.shape[0]):
                        filtered_data[i] = signal.convolve(filtered_data[i], win, mode='same') / sum(win)
                        # Hann window generates edge artifacts due to zero-padding. Retain unfiltered data at edges.
                        filtered_data[i, :hann] = original_data[i, :hann]
                        filtered_data[i, filtered_data.shape[1] - hann:] = original_data[i, filtered_data.shape[1] - hann:]
            
            return filtered_data

        # Apply filtering to current data
        filtered_current = apply_filters(self.current_data, self.current_data)
        
        # Apply same filtering to voltage data if requested and available
        filtered_voltage = None
        if self.voltage_data is not None and apply_to_voltage:
            filtered_voltage = apply_filters(self.voltage_data, self.voltage_data)

        return Trace(filtered_current, sampling_interval=self.sampling, current_unit=self.current_unit, filename=self.filename,
                    voltage_data=filtered_voltage, voltage_unit=self.voltage_unit,
                    concatenate_sweeps=getattr(self, 'concatenate_sweeps', True))

    def resample(self, sampling_frequency: float=None):
        ''' Resamples the data trace to the given frequency 
        
        Parameters
        ----------
        sampling_frequency: float
            Sampling frequency in Hz of the output data
            
        Returns
        -------
        Trace
            A resampled Trace object
        '''
        if sampling_frequency is None:
            return self

        if self.current_data is None:
            raise ValueError("No data to resample")

        resampling_factor = np.round(self.sampling_rate / sampling_frequency, 2)
        new_sampling_interval = self.sampling * resampling_factor
        
        if self.current_data.ndim == 1:
            # Handle 1D data (concatenated sweeps)
            resampled_current = signal.resample(self.current_data, int(self.current_data.shape[0]/resampling_factor))
            
            # Also resample voltage data if available
            resampled_voltage = None
            if self.voltage_data is not None:
                resampled_voltage = signal.resample(self.voltage_data, int(self.voltage_data.shape[0]/resampling_factor))
        else:
            # Handle 2D data (separate sweeps)
            new_length = int(self.current_data.shape[1] / resampling_factor)
            resampled_current = np.zeros((self.current_data.shape[0], new_length))
            
            for i in range(self.current_data.shape[0]):
                resampled_current[i] = signal.resample(self.current_data[i], new_length)
            
            # Also resample voltage data if available
            resampled_voltage = None
            if self.voltage_data is not None:
                resampled_voltage = np.zeros((self.voltage_data.shape[0], new_length))
                for i in range(self.voltage_data.shape[0]):
                    resampled_voltage[i] = signal.resample(self.voltage_data[i], new_length)

        return Trace(resampled_current, sampling_interval=new_sampling_interval, current_unit=self.current_unit, 
                    filename=self.filename, voltage_data=resampled_voltage, voltage_unit=self.voltage_unit,
                    concatenate_sweeps=getattr(self, 'concatenate_sweeps', True))


    def copy(self):
        """
        Create a deep copy of the Trace object.
        
        Returns
        -------
        Trace
            A new Trace object with copied data arrays and attributes.
            All data is deep copied to prevent unintended modifications.
        """
        import copy as copy_module
        return copy_module.deepcopy(self)



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


def time_to_index(t, sampling_freq, time_unit='ms'):
    """
    Convert time in milliseconds to index in the sweep, based on the sampling frequency.
    """
    if time_unit in ['ms', 'milliseconds']:
        return int(t * sampling_freq / 1000)
    elif time_unit in ['s', 'seconds']:
        return int(t * sampling_freq)



###############################
# Plotting functions
###############################
def update_plot_defaults():
    plt.rcParams.update({
                    'font.size': 12,
                    # 'figure.figsize': [10.0, 3.0],
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


def plot_traces(time, current_traces, voltage_traces=None, marker_1=None, marker_2=None, ax=None, height_ratios=(3, 1)):
    """Plot current and optionally voltage traces with markers."""
    
    # Determine number of subplots
    has_voltage = voltage_traces is not None
    
    # Create or validate axes
    if ax is None:
        if has_voltage:
            fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=False, height_ratios=height_ratios)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax = [ax]  # Make it a list for consistent indexing
    else:
        # Ensure ax is a list for consistent indexing
        if not isinstance(ax, (list, np.ndarray)):
            ax = [ax]
    
    # Plot current traces
    ax[0].plot(time, current_traces.T, color='black', linewidth=0.8)
    ax[0].set_ylabel('Current (pA)')
    ax[0].set_xlabel('Time')
    
    # Plot voltage traces if provided
    if has_voltage:
        ax[1].set_prop_cycle(color=plt.cm.viridis(np.linspace(0, 1, voltage_traces.shape[0])))
        ax[1].plot(time, voltage_traces.T, color='black', linewidth=0.5)
        ax[1].set_ylabel('Voltage (mV)')
    
    # Add markers to all active subplots
    active_axes = ax[:2] if has_voltage else ax[:1]
    
    for marker, label in [(marker_1, 'marker 1'), (marker_2, 'marker 2')]:
        if marker is not None:
            for subplot in active_axes:
                ylims = subplot.get_ylim()
                subplot.vlines(marker, *ylims, color='red', linestyle='-', linewidth=0.5)
                subplot.text(marker, ylims[1], label, fontsize=10, color='red', ha='center', va='bottom')
    
    # Set x-axis limits for all active subplots
    for subplot in active_axes:
        subplot.set_xlim(time[0], time[-1])
    
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