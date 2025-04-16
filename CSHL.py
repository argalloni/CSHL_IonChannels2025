def IV(f,vtimes,btimes,ptimes,plot=True,
       xcoord=(0.9, 0.4),ycoord=(0.45, 0.9)):
    '''
    Takes a filename : f
    ALL TIMES IN SECONDS
    voltage step window (vtimes): (v_start,v_end)
    baseline current window (btimes): (i_start,i_end)
    peak current window (ptimes): (i_start,i_end)
    plot : boolean to show or not the plot
    xcoord and ycoord are coordinate for the axis labels
    
    returns the peak amplitudes and voltage steps values.
    '''
    # Extract the sweeps,time and sampling rate:
    swps, swp_time, sr = get_sweeps(f)
    # Extract the start and end window times:
    v_start,v_end = vtimes
    b_start,b_end = btimes
    p_start,p_end = ptimes
    # extract the voltage steps:
    voltage_trace = np.mean(swp_window(swps,v_start,v_end,sr,channel=1),axis=1)
    # extract baseline current:
    baseline_current = np.mean(swp_window(swps,b_start,b_end,sr,channel=0),axis=1)
    # extract peak current:
    peak_window = swp_window(swps,p_start,p_end,sr,channel=0)
    peak_response = peak(peak_window)
    # and normalise over the baseline current:
    peak_response -= baseline_current
    # plot the result:
    if plot==True:
        fig,ax = plt.subplots()
        ax.plot(voltage_step,peak_response,'-o',alpha=0.9)
        IV_style(ax,xcoord=xcoord,ycoord=ycoord)
        plt.show()
    
    return voltage_step,peak_response

def IV_style(ax,
            xcoord=(0.9, 0.4),
            ycoord=(0.45, 0.9)):
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.set_xlabel('V (mV)')
    ax.set_ylabel('I/Cm (pA/pF)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_label_coords(xcoord[0], xcoord[1]) 
    ax.yaxis.set_label_coords(ycoord[0], ycoord[1])
    # Customize ticks to remove the 0 ticks and labels
    xticks = [tick for tick in ax.get_xticks() if tick != 0]
    yticks = [tick for tick in ax.get_yticks() if tick != 0]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

def spike_detect(volt, v_dt, dur,
                 thr=-10, inf_bnd=3, sup_bnd=500,
                 ref_t=1e-3, sigma=1e-3):
    deriv = np.diff(volt)
    is_spk = (deriv[:-1]<2) * (deriv[1:]>=2)
    spikes = np.where(is_spk)[0]
    refact = int(ref_t/v_dt)
    ref = 0
    sspk = []
    for st in spikes:
        if st<ref:
            continue
        else:
            spk = volt[st:st+refact]
        if spk.max() < thr:
            continue
        else:
            sspk.append(st)
            ref = st+refact
    sspk = np.array(sspk)
    stimes = sspk*v_dt
    isi = np.diff(stimes)
    inst_frq = 1/isi
    smth_frq = smooth(stimes[1:], sigma, inst_frq)
    out_frq = (inf_bnd>smth_frq) | (smth_frq>sup_bnd)
    if np.any(out_frq):
        stop = np.where(out_frq)[0][0] - 1
        dur = (sspk[stop]+refact)*v_dt
        sspk = sspk[:stop]
        stimes = sspk*v_dt
        isi = np.diff(stimes)
        inst_frq = 1/isi
    is_spk = np.zeros(int(dur/v_dt))
    is_spk[sspk]=True
    return [sspk,stimes], inst_frq, is_spk, dur

