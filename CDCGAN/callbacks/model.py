# File management
import os
import sys
import glob

# Data science
import tensorflow as tf

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Classes
from utils.file_management import FileManagement


class SaveLoadGeneratorDiscriminatorCallback(tf.keras.callbacks.Callback):
    '''Custom callback to save both generator and discriminator models'''
    def __init__(self, model_list: list[tf.keras.Model], 
                 model_ckpt_dir: str, 
                 num_ckpt_to_save: int, 
                 save_all=bool,
                 verbose=False):
        '''Initialize model list

        Args:
            model_list (list[tf.keras.Model]): list of models you want to save
            model_ckpt_dir (str): path to the model checkpoint where the modes is saved
            num_ckpt_to_save (int): latest number of mmodels to save            
            save_all (bool, optional): Save all model or only a number, Defaults to False.
            verbose (bool): printing stuff
        '''
        super().__init__()
        self.models_list = model_list
        self.model_ckpt_dir = model_ckpt_dir
        self.num_ckpt_to_save = num_ckpt_to_save
        self.save_all = save_all
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        '''Saving model to its folder'''
        # Looping through each model in the list
        for model in self.models_list:
            # Model name
            model_name = model.name

            # Model dir in checkpoints
            model_dir = os.path.join(self.model_ckpt_dir, model_name)

            model_save_name = f'{model.name}_epoch_{epoch + 1}.h5'

            # Path to the model dir in the checkpoint dir
            model_save_path = os.path.join(model_dir, model_save_name)
            
            # Save model
            model.save(model_save_path)
            if self.verbose:
                print(f'Saved model: {model_save_path}')

            # If we dont want to save all checkpoints then we keep only the desired number of models
            if not self.save_all:
                # Remove the checkpoints further back
                list_of_model_ckpts = glob.glob(model_dir + '/*')
                list_of_model_ckpts = sorted(list_of_model_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                for model_file_path in list_of_model_ckpts[:-self.num_ckpt_to_save]:
                    os.remove(model_file_path)
                    if self.verbose:
                        print(f'Removing model: {model_file_path} ...')

    def on_train_begin(self, logs=None):
        '''Creating folders if training from scratch else load weights to model'''
        # Looping through each model in the list
        for model in self.models_list:
            # Model name
            model_name = model.name

            # Path to the model dir in the model checkpoint dir
            model_dir = os.path.join(self.model_ckpt_dir, model_name)
            
            # Create new directory if training from scratch
            if not os.path.exists(model_dir):
                # Create a new directory for that model in the checkpoints dir
                os.mkdir(model_dir)
                print(f'Creating folder: {model_dir} ...')
            else: # Else load models latest model from previous training
                # List all model files in its directory
                model_files = glob.glob(model_dir + '/*')
                
                # Load generator only if it exist
                if model_files:
                    model_files = FileManagement.sort_files_by_number(model_files)
                    
                    # Load the latest generator file
                    model_file = model_files[-1]
                    model.load_weights(filepath=model_file)
                    print(f'Loading {model_name} model: {model_file} ...')


