import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
import argparse
import h5py as h5
import utils
from ABCNet import ABCNet, SWD
import fastjet
import awkward as ak
import itertools


def GetMET(parts):
    '''
    Inputs:
    data of shape (N,npart,4)
    Outputs:
    MET pt, MET phi: (N,2)
    '''

    MET = -np.sum(parts,1)
    MET_pt = np.sqrt(MET[:,0]**2+MET[:,1]**2)
    MET_phi = np.arctan2(MET[:,1],MET[:,0])
    return np.stack((MET_pt,MET_phi),-1)

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    parser = argparse.ArgumentParser()
        
    parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/PU', help='Folder containing data and MC files')
    parser.add_argument('--nevts', type=int,default=-1, help='Number of events to load')
    parser.add_argument('--config', default='config.json', help='Config file with training parameters')
    
    flags = parser.parse_args()
    dataset_config = utils.LoadJson(flags.config)
    preprocessing = utils.LoadJson(dataset_config['PREPFILE'])
    
    dataset = dataset_config['FILES'][0]
    NPART=dataset_config['NPART']
    checkpoint_folder = '../checkpoints_{}'.format(dataset_config['CHECKPOINT_NAME'])
    data = utils.EvalLoader(os.path.join(flags.data_folder,dataset),flags.nevts)
    inputs,outputs = ABCNet(npoint=NPART,nfeat=dataset_config['SHAPE'][2])
    model = Model(inputs=inputs,outputs=outputs)
    model.load_weights('{}/{}'.format(checkpoint_folder,'checkpoint'))


    abcnet_weights = model.predict(utils.ApplyPrep(preprocessing,data['pu_part'][:,:NPART]),batch_size=10)
    abcnet_weights[abcnet_weights<1e-3]=0

    #abcnet_weights = model.predict(data['pu_part'][:,:NPART],batch_size=10)
    puppi_weights = data['pu_part'][:,:NPART,-3]
    def _convert_kinematics(data,is_gen=False):
        four_vec = data[:,:,:3]
        #eta,phi,pT,E
        #pt and energy conversion from log transformation
        # if not is_gen:
        #     four_vec[:,:,2][four_vec[:,:,2]!=0] = 10**(four_vec[:,:,2][four_vec[:,:,2]!=0])
        #     four_vec[:,:,3][four_vec[:,:,3]!=0] = 10**(four_vec[:,:,3][four_vec[:,:,3]!=0])
            
        #convert to cartesian coordinates (px,py,pz,E)
        cartesian = np.zeros(four_vec.shape,dtype=np.float32)
        cartesian[:,:,0] += np.abs(four_vec[:,:,2])*np.cos(four_vec[:,:,1])
        cartesian[:,:,1] += np.abs(four_vec[:,:,2])*np.sin(four_vec[:,:,1])
        cartesian[:,:,2] += np.abs(four_vec[:,:,2])*np.ma.sinh(four_vec[:,:,0]).filled(0)
        # cartesian[:,:,3] = four_vec[:,:,3]
        #print(cartesian)
        return cartesian

    nopu_set = _convert_kinematics(data['nopu_part'])
    gen_set =  _convert_kinematics(data['gen_part'],is_gen=True)
    abc_set = _convert_kinematics(data['pu_part'][:,:NPART])*(abcnet_weights)
    puppi_set = _convert_kinematics(data['pu_part'][:,:NPART])*np.expand_dims(puppi_weights,-1)

    print(abc_set.dtype)
    del data['nopu_part'], data['pu_part']
    # abc_set = _convert_kinematics(data['pu_part'][:,:,:4]*abcnet_weights)



    # pu_mask = np.sum(pu_set[:,:,2]!=0,1)<NPART #avoid events that are truncated
    # puppi_set = puppi_set[pu_mask]
    # abc_set = abc_set[pu_mask]
    # nopu_set=nopu_set[pu_mask]
    # gen_set = gen_set[pu_mask]
    

    dict_dataset = {}
    
    dict_dataset['MET_gen'] = GetMET(gen_set)
    dict_dataset['MET_nopu'] = GetMET(nopu_set)
    dict_dataset['MET_puppi'] = GetMET(puppi_set)
    dict_dataset['MET_abc'] = GetMET(abc_set)
    
    def _cluster(data):
        jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
        #dumb conversion for inputs
        events = []
        for datum in data:
            events.append([{"px": part[0], "py": part[1], "pz": part[2], "E": (part[0]**2+part[1]**2+part[2]**2)**0.5} for part in datum if np.abs(part[0])>0] )
            
        array = ak.Array(events)
        cluster = fastjet.ClusterSequence(array, jetdef)
        jets = cluster.inclusive_jets(min_pt=10)
        jets["pt"] = np.sqrt(jets["px"]**2 + jets["py"]**2)
        jets["phi"] = np.arctan2(jets["py"],jets["px"])
        jets["eta"] = np.arcsinh(jets["pz"]/jets["pt"])
        jets=fastjet.sorted_by_pt(jets) 
        return jets[:,::-1]

    sets = {'nopu_jet':nopu_set,'abc_jet':abc_set,'puppi_jet':puppi_set,'gen_jet':gen_set}

    def _dict_data(jets,njets):        
        cluster = _cluster(jets)
        dataset = np.zeros((len(cluster.pt.to_list()),njets,4),dtype=np.float32)

        dataset[:,:,0]+=np.array(list(itertools.zip_longest(*cluster.pt.to_list(), fillvalue=0))).T[:,:njets]
        dataset[:,:,1]+=np.array(list(itertools.zip_longest(*cluster.eta.to_list(), fillvalue=0))).T[:,:njets]
        dataset[:,:,2]+=np.array(list(itertools.zip_longest(*cluster.phi.to_list(), fillvalue=0))).T[:,:njets]
        dataset[:,:,3]+=np.array(list(itertools.zip_longest(*cluster.E.to_list(), fillvalue=0))).T[:,:njets]
        return dataset
    
    for dset in sets:
        print(dset)
        dict_dataset[dset] = _dict_data(sets[dset],njets=10)
        
    with h5.File(os.path.join(flags.data_folder,"JetInfo_{}_{}".format(dataset_config['CHECKPOINT_NAME'],dataset)),"w") as h5f:
        for key in dict_dataset:
            dset = h5f.create_dataset(key, data=dict_dataset[key])
            
