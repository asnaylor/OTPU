import json, yaml
import os
import h5py as h5
import horovod.tensorflow.keras as hvd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick

def split_data(data,nevts,frac=0.8):
    data = data.shuffle(nevts)
    train_data = data.take(int(frac*nevts)).repeat()
    test_data = data.skip(int(frac*nevts)).repeat()
    return train_data,test_data

line_style = {
    'Delphes':'dotted',
    'ABCNet':'-',
}

colors = {
    'Delphes':'black',
    'ABCNet':'#7570b3',
}


def SetStyle():
    from matplotlib import rc
    rc('text', usetex=True)

    import matplotlib as mpl
    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('legend', fontsize=15)

    # #
    mpl.rcParams.update({'font.size': 19})
    #mpl.rcParams.update({'legend.fontsize': 18})
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams.update({'xtick.labelsize': 18}) 
    mpl.rcParams.update({'ytick.labelsize': 18}) 
    mpl.rcParams.update({'axes.labelsize': 18}) 
    mpl.rcParams.update({'legend.frameon': False}) 
    mpl.rcParams.update({'lines.linewidth': 2})
    
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.set_style(hep.style.CMS)
    hep.style.use("CMS") 

def SetGrid(ratio=True):
    fig = plt.figure(figsize=(9, 9))
    if ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
        gs.update(wspace=0.025, hspace=0.1)
    else:
        gs = gridspec.GridSpec(1, 1)
    return fig,gs



def PlotRoutine(feed_dict,xlabel='',ylabel='',reference_name='Geant4'):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"
    
    fig,gs = SetGrid() 
    ax0 = plt.subplot(gs[0])
    plt.xticks(fontsize=0)
    ax1 = plt.subplot(gs[1],sharex=ax0)

    for ip,plot in enumerate(feed_dict.keys()):
        if 'steps' in plot or 'r=' in plot:
            ax0.plot(np.mean(feed_dict[plot],0),label=plot,marker=line_style[plot],color=colors[plot],lw=0)
        else:
            ax0.plot(np.mean(feed_dict[plot],0),label=plot,linestyle=line_style[plot],color=colors[plot])
        if reference_name!=plot:
            ratio = 100*np.divide(np.mean(feed_dict[reference_name],0)-np.mean(feed_dict[plot],0),np.mean(feed_dict[reference_name],0))
            #ax1.plot(ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)
            if 'steps' in plot or 'r=' in plot:
                ax1.plot(ratio,color=colors[plot],markeredgewidth=1,marker=line_style[plot],lw=0)
            else:
                ax1.plot(ratio,color=colors[plot],linewidth=2,linestyle=line_style[plot])
                
        
    FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0)
    ax0.legend(loc='best',fontsize=16,ncol=1)

    plt.ylabel('Difference. (%)')
    plt.xlabel(xlabel)
    plt.axhline(y=0.0, color='r', linestyle='--',linewidth=1)
    plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
    plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
    plt.ylim([-50,50])

    return fig,ax0

class ScalarFormatterClass(mtick.ScalarFormatter):
    #https://www.tutorialspoint.com/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot
    def _set_format(self):
        self.format = "%1.2f"


def FormatFig(xlabel,ylabel,ax0):
    #Limit number of digits in ticks
    # y_loc, _ = plt.yticks()
    # y_update = ['%.1f' % y for y in y_loc]
    # plt.yticks(y_loc, y_update) 
    ax0.set_xlabel(xlabel,fontsize=20)
    ax0.set_ylabel(ylabel)
        

def WriteText(xpos,ypos,text,ax0):

    plt.text(xpos, ypos,text,
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax0.transAxes, fontsize=25, fontweight='bold')


def HistRoutine(feed_dict,xlabel='',ylabel='',reference_name='Geant4',logy=False,binning=None,label_loc='best'):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"
    
    fig,gs = SetGrid() 
    ax0 = plt.subplot(gs[0])
    plt.xticks(fontsize=0)
    ax1 = plt.subplot(gs[1],sharex=ax0)

    
    if binning is None:
        binning = np.linspace(np.quantile(feed_dict[reference_name],0.0),np.quantile(feed_dict[reference_name],1),10)
        
    xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
    reference_hist,_ = np.histogram(feed_dict[reference_name],bins=binning,density=True)
    
    for ip,plot in enumerate(feed_dict.keys()):
        if 'steps' in plot or 'r=' in plot:
            dist,_ = np.histogram(feed_dict[plot],bins=binning,density=True)
            ax0.plot(xaxis,dist,color=colors[plot],marker=line_style[plot],ms=10,lw=0,markeredgewidth=3,label=plot)
            #dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot,marker=line_style[plot],color=colors[plot],density=True,histtype="step")
        else:
            dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot,linestyle=line_style[plot],color=colors[plot],density=True,histtype="step")
            
        if reference_name!=plot:
            ratio = 100*np.divide(reference_hist-dist,reference_hist)
            if 'steps' in plot or 'r=' in plot:
                ax1.plot(xaxis,ratio,color=colors[plot],marker=line_style[plot],ms=10,lw=0,markeredgewidth=3)
            else:
                ax1.plot(xaxis,ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)
        
    ax0.legend(loc=label_loc,fontsize=16,ncol=1)        
    FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0) 

    if logy:
        ax0.set_yscale('log')
    
    plt.ylabel('Difference. (%)')
    plt.xlabel(xlabel)
    plt.axhline(y=0.0, color='r', linestyle='-',linewidth=1)
    plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
    plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
    plt.ylim([-50,50])

    return fig,ax0


def DataLoader(file_name,nevts):
    rank = hvd.rank()
    size = hvd.size()

    with h5.File(file_name,"r") as h5f:
        pu = h5f['pu_part'][rank:int(nevts):size].astype(np.float32)
        nopu = h5f['nopu_part'][rank:int(nevts):size].astype(np.float32)
    return pu,nopu
        

def LoadJson(file_name):
    import json,yaml
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))




def Preprocess(name,raw_data):
    '''Preprocess the data'''
    if 'Eta' in name or 'Phi' in name or 'PuppiW' in name or 'Charge' in name:
        #no modification
        return np.array(raw_data)
    elif 'PT' in name or 'E' in name:
        return np.ma.log10(raw_data).filled(0)
    else:
        return np.sign(raw_data)*np.ma.log10(np.abs(raw_data)).filled(0)/10.0
    
if __name__ == "__main__":
    #Preprocessing of the input files: conversion to cartesian coordinates + zero-padded mask generation
    import uproot3 as uproot
    hvd.init()

    features = ['Eta','Phi','PT','E','Eem','Ehad','D0','DZ','PuppiW','PID','Charge']
    pu_features = ["pu_pfs_{}".format(feat) for feat in features]
    nopu_features = ["nopu_pfs_{}".format(feat) for feat in features]
    genpart_branches = ['Eta','Phi','PT','E','Charge','PID']
    gen_info = ["nopu_gen_{}".format(gen) for gen in genpart_branches]
    high_level = ['nopu_genmet_MET','nopu_genmet_Phi']
    
    base_path = '/global/cfs/cdirs/m3929/PU/'
    file_list = ['VBFHinv_outfile_{}.root'.format(i) for i in range(2,11)]
    merged_file = {}

    print("merging files")
    for sample in file_list:
        file_path = os.path.join(base_path,sample)
        temp_file = uproot.open(file_path)['events']
        
        for feat in gen_info + nopu_features + pu_features + high_level:
            if feat in merged_file:
                merged_file[feat] = np.concatenate([merged_file[feat],temp_file[feat].array()],0)
            else:
                merged_file[feat] = temp_file[feat].array()
    del temp_file
    print("Preprocessing")

    def _merger(features,shape):
        array=[]
        for feat in features:
            array.append(merged_file[feat])
        return np.reshape(array,shape).astype(np.float32)

    
    high_array = _merger(high_level,shape=(-1,len(high_level)))
    gen_array = _merger(gen_info,shape=(-1,1000,len(gen_info)))

    
    #Standardize training data prior to training
    for feat in nopu_features + pu_features:
        merged_file[feat] = Preprocess(feat,merged_file[feat])
        #print(feat,np.min(merged_file[feat]),np.max(merged_file[feat]))

    nopu_array = _merger(nopu_features,shape=(-1,9000,len(nopu_features)))
    pu_array = _merger(pu_features,shape=(-1,9000,len(pu_features)))

    with h5.File(os.path.join(base_path,'VBFHinv.h5'),'w') as fh5:
        dset = fh5.create_dataset('high_level', data=high_array)
        dset = fh5.create_dataset('gen_part', data=gen_array)
        dset = fh5.create_dataset('pu_part', data=pu_array)
        dset = fh5.create_dataset('nopu_part', data=nopu_array)
                
    
