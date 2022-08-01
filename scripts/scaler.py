import numpy as np
import os
import h5py as h5
import utils



data_folder = '/pscratch/sd/v/vmikuni/PU'
dataset = 'DiJet_raw.h5'

data_ = utils.EvalLoader(os.path.join(data_folder,dataset),-1)['nopu_part']
shape = data_.shape

data_flat = data_.reshape((-1,shape[-1]))
mask = data_flat!=0
mean = np.average(data_flat,axis=0,weights=mask) -1

data_dict = {
    'mean':mean.tolist(),
    'std':np.sqrt(np.average((data_flat - mean+1)**2,axis=0,weights=mask)).tolist()
}

json_file = 'preprocessing_raw.json'
utils.SaveJson(json_file,data_dict)

