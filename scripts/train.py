import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
import horovod.tensorflow.keras as hvd
from tensorflow.keras.models import Model
import argparse
import h5py as h5
import utils
from ABCNet import ABCNet, SWD


if __name__ == '__main__':
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')



    parser = argparse.ArgumentParser()
        
    parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/PU/', help='Folder containing data and MC files')
    parser.add_argument('--nevts', type=float,default=-1, help='Number of events to load')
    parser.add_argument('--config', default='config.json', help='Config file with training parameters')
    parser.add_argument('--frac', type=float,default=0.8, help='Fraction of total events used for training')
    parser.add_argument('--load', action='store_true', default=False,help='Load pretrained weights to continue the training')
    flags = parser.parse_args()
    dataset_config = utils.LoadJson(flags.config)

    data = []
    label = []
    NSWD = dataset_config['NSWD'] #SWD is calculated considering only NSWD features
    for dataset in dataset_config['FILES']:
        data_,label_ = utils.DataLoader(
            os.path.join(flags.data_folder,dataset),flags.nevts)
        
        data.append(data_)
        label.append(label_)
        

    data = np.reshape(data,dataset_config['SHAPE'])
    data_size = data.shape[0]
    label = np.reshape(label,dataset_config['SHAPE'])[:,:,:NSWD]


    dataset = tf.data.Dataset.from_tensor_slices((data,np.concatenate([data[:,:,:NSWD],label],-1)))
    train_data, test_data = utils.split_data(dataset,data_size,flags.frac)
    del dataset, data, label

    BATCH_SIZE = dataset_config['BATCH']
    LR = float(dataset_config['LR'])
    NUM_EPOCHS = dataset_config['MAXEPOCH']
    EARLY_STOP = dataset_config['EARLYSTOP']


    
    inputs,outputs = ABCNet(npoint=dataset_config['SHAPE'][1],nfeat=dataset_config['SHAPE'][2])
    model = Model(inputs=inputs,outputs=outputs)
    opt = keras.optimizers.Adam(learning_rate=LR)
    opt = hvd.DistributedOptimizer(
        opt, average_aggregated_gradients=True)

    model.compile(loss=SWD,
                  optimizer=opt,experimental_run_tf_function=False)
    if hvd.rank()==0:
        print(model.summary())

    
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),            
        ReduceLROnPlateau(patience=30, factor=0.5,
                          min_lr=1e-8,verbose=hvd.rank()==0),
        EarlyStopping(patience=EARLY_STOP,restore_best_weights=True),
    ]

    
    history = model.fit(
        train_data.batch(BATCH_SIZE),
        epochs=NUM_EPOCHS,
        steps_per_epoch=int(data_size*flags.frac/BATCH_SIZE),
        validation_data=test_data.batch(BATCH_SIZE),
        validation_steps=int(data_size*(1-flags.frac)/BATCH_SIZE),
        verbose=1 if hvd.rank()==0 else 0,
        callbacks=callbacks
    )




    if hvd.rank()==0:
        checkpoint_folder = '../checkpoints_{}'.format(dataset_config['CHECKPOINT_NAME'])
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        os.system('cp ABCNet.py {}'.format(checkpoint_folder)) # bkp of model def
        os.system('cp {} {}'.format(flags.config,checkpoint_folder)) # bkp of config file
        model.save_weights('{}/{}'.format(checkpoint_folder,'checkpoint'),save_format='tf')
