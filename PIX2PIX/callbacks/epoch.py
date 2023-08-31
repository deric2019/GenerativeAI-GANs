# File management
import os
import glob

# Data science
import tensorflow as tf
import numpy as np

class EpochCallback(tf.keras.callbacks.Callback):
    '''Custom callback for saving epoch number '''
    def __init__(self, epoch_ckpt_dir, epoch_i_npy_file):
        '''Initializing epoch number
        Args: 
            epoch_ckpt_dir: path to folder where the file stores
            epoch_i_npy_file: filepath with filepattern to the npy file'''
        super().__init__()
        self.epoch_i = 0

        self.epoch_ckpt_dir = epoch_ckpt_dir
        self.epoch_i_npy_file = epoch_i_npy_file

    def generate_epoch_name(self):
        '''Generate name to numpy file'''
        # at_epoch_{}.npy
        return os.path.join(self.epoch_ckpt_dir, f'at_epoch_{self.epoch_i}.npy')
    
    def on_train_begin(self, logs=None):
        '''Creating directory if it does not already exist, else load epoch number'''
        # If file already exists then load the latest epoch number, else create a new one
        list_of_files = glob.glob(self.epoch_i_npy_file)
        if list_of_files:
            self.epoch_i = np.load(list_of_files[0])
        else: 
            print('Starting training from scratch ...' )

            print(f'Creating folder: {self.epoch_ckpt_dir} ...')
            os.mkdir(self.epoch_ckpt_dir)

            print(f'Creating file: {self.generate_epoch_name()} ...')
            np.save(self.generate_epoch_name(), self.epoch_i)

    def on_epoch_end(self, epoch, logs=None):
        '''Increase epoch by 1'''
        # Remove the latest epoch number file
        os.remove(glob.glob(self.epoch_i_npy_file)[0])

        # Save npy.file
        self.epoch_i += 1
        np.save(file=self.generate_epoch_name(), arr=self.epoch_i)
