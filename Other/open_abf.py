import numpy as np
import pyabf
import matplotlib.pyplot as plt
import matplotlib as mp

def get_protocol(files):
    protocols = []
    for f in files:
        rec = pyabf.ABF(f)
        protocols.append(rec.protocol)
    return protocols

def files_from_protocol(prt_name,files):
    protocols = get_protocol(files)
    prt_file = []
    for i,prt in enumerate(protocols):
        if (prt_name in prt):
            prt_file.append(files[i])
    return prt_file

def plot_swps(file,legend=False,
              cmap = mp.colormaps['viridis']):
    rec = pyabf.ABF(file)
    prt = rec.protocol
    fig,ax = plt.subplots(2,sharex=True)
    ax[0].set_title(prt+' ('+ file.split('/')[-1]+')')
    for i,swpNB in enumerate(rec.sweepList):
        rec.setSweep(swpNB)
        color = cmap(i/len(rec.sweepList))[:-1]
        ax[0].plot(rec.sweepX,rec.sweepY,color=color)
        ax[1].plot(rec.sweepX,rec.sweepC,label=f"sweep:{swpNB}",color=color)
    # labels and legends
    ax[1].set_xlabel(rec.sweepLabelX)
    ax[0].set_ylabel(rec.sweepLabelY)
    ax[1].set_ylabel(rec.sweepLabelC)
    # ax[0].set_xticks([])
    for i in range(2):
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
    if legend:
        ax[1].legend(loc="right")
    fig.tight_layout()
    plt.show()

def plot_swps_DO(file, DOchans,legend=False):
    rec = pyabf.ABF(file)
    prt = rec.protocol
    nb_plot = 2+len(DOchans)
    fig,ax = plt.subplots(nb_plot,sharex=True)
    ax[0].set_title(prt+' ('+ file.split('/')[-1]+')')
    for swpNB in rec.sweepList:
        rec.setSweep(swpNB)
        ax[0].plot(rec.sweepX,rec.sweepY)
        ax[1].plot(rec.sweepX,rec.sweepC,label=f"sweep:{swpNB}")
        for i,chn in enumerate(DOchans):
            ax[i+2].plot(rec.sweepX,rec.sweepD(chn),label=f"sweep:{swpNB}")
    # labels and legends
    ax[nb_plot-1].set_xlabel(rec.sweepLabelX)
    ax[0].set_ylabel(rec.sweepLabelY)
    ax[1].set_ylabel(rec.sweepLabelC)
    for i,chn in enumerate(DOchans):
        ax[i+2].set_ylabel(f"DO {chn}")
    for i in range(nb_plot):
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
    if legend:
        ax[1].legend(loc="right")
    fig.tight_layout()
    plt.show()

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

def get_sweeps_DO(f,DOchans):
    rec = pyabf.ABF(f)
    swps = []
    DO = []
    for swpNB in rec.sweepList:
        rec.setSweep(swpNB)
        swp = [rec.sweepY,rec.sweepC]
        do = []
        for chn in DOchans:
            do.append(rec.sweepD(chn))
        DO.append(do)
        swps.append(swp)
    swps = np.array(swps)
    swp_time = rec.sweepX
    dt = swp_time[1] 
    return swps,np.array(DO), swp_time, 1/dt

def swp_window(swps,start,end,sr,channel=0):
    i_start = int(start * sr)
    i_end = int(end * sr)
    return swps[:,channel,i_start:i_end]
