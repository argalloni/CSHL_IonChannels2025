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

