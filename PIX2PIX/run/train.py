# File management 
import os 
import sys 
import glob

# Configuration
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import Config

# Data science libraries
import numpy as np
import tensorflow as tf



# Classes
from dataloader.data_loader import DataLoader 
from utils.file_management import FileManagement
from models.pix2pix import PIX2PIX

from callbacks.epoch import EpochCallback
from callbacks.image import GenerateSaveImagesCallback
from callbacks.model import SaveLoadGeneratorDiscriminatorCallback


'''
Main functions for the project, can also be seen as modes of the project
'''
class Train():
    '''Class consisting of function acting on the model'''
    def train():
        '''Train the our model'''
        # --------------
        # Load dataset
        # --------------
        dataset_dict = DataLoader.load_dataset()
        
        # -----------------------------
        # Create instance of cyclegan
        # ------------------------------
        loss_to_optimize = Config.Settings.loss_to_optimize
        pix2pix_o = PIX2PIX(loss_to_optimize=loss_to_optimize)

        # -----------------------------
        # Optimizers and loss functions
        # ------------------------------
        # Compile the model
        lr, beta_1 = Config.ModelParam.learning_rate, Config.ModelParam.learning_rate
        pix2pix_o.compile(
            generator_optimizer=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1),
            discriminator_optimizer=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1),
            adv_loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            l1_loss_fn=tf.keras.losses.MeanAbsoluteError(), 
            run_eagerly=False
        ) 
        
        # --------------------
        # Callbacks
        # --------------------
        ### Epoch
        epoch_ckpt_dir = Config.Path.Checkpoints.epoch_ckpt_dir
        epoch_i_npy_file = Config.Path.Checkpoints.epoch_i_npy_file
        epoch_ckpt_callback = EpochCallback(epoch_ckpt_dir, epoch_i_npy_file)

        ### Models
        model_ckpt_dir = Config.Path.Checkpoints.model_ckpt_dir
        num_ckpt_to_save = Config.Settings.num_ckpt_to_save
        save_all_models = Config.Settings.save_all_models
        model_ckpt_callback = SaveLoadGeneratorDiscriminatorCallback(model_list=[pix2pix_o.generator, 
                                                                               pix2pix_o.discriminator],
                                                                                model_ckpt_dir=model_ckpt_dir,
                                                                                num_ckpt_to_save=num_ckpt_to_save,
                                                                                save_all=save_all_models)
        
        ### Images
        img_during_training_ckpt_dir = Config.Path.Checkpoints.img_during_training_ckpt_dir
        save_every_n_epochs = Config.Settings.save_image_every_n_epochs
        images_callback = GenerateSaveImagesCallback(generator=pix2pix_o.generator, 
                                                     dataset = dataset_dict['val'], 
                                                     img_during_training_ckpt_dir=img_during_training_ckpt_dir,
                                                     save_every_n_epochs=save_every_n_epochs
                                                     )
        
        csvlog_ckpt_callback = tf.keras.callbacks.CSVLogger(filename=Config.Path.Checkpoints.csvlog_log_file, append=True)
        
        callback_list = [epoch_ckpt_callback, model_ckpt_callback, images_callback, csvlog_ckpt_callback]

        # -----------------------------
        # Initializing of checkpoints
        # -----------------------------
        # Also returns the latest epoch
        epoch_i = Train.initializing_checkpoints_folders()

        # --------------------
        # Train model
        # --------------------
        pix2pix_o.fit(dataset_dict['train'], epochs=Config.ModelParam.epochs, initial_epoch=epoch_i, callbacks=callback_list)


    def initializing_checkpoints_folders():
        '''Creating the checkpoint and csv logger dir if it dos not already exists'''
        # Checkpoint dir
        ckpt_dir = Config.Path.Checkpoints.dir
        FileManagement.create_folder_it_not_already_exists(ckpt_dir)

        model_ckpt_dir = Config.Path.Checkpoints.model_ckpt_dir
        FileManagement.create_folder_it_not_already_exists(model_ckpt_dir)
        
        # Create an csvlog dir 
        csvlog_ckpt_dir = Config.Path.Checkpoints.csvlog_ckpt_dir
        FileManagement.create_folder_it_not_already_exists(csvlog_ckpt_dir)

        ### Create an image directory if it does not already exists
        img_during_training_ckpt_dir = Config.Path.Checkpoints.img_during_training_ckpt_dir
        FileManagement.create_folder_it_not_already_exists(img_during_training_ckpt_dir)

        # Resume epoch number have trained before
        epoch_ckpt_dir = Config.Path.Checkpoints.epoch_ckpt_dir
        epoch_i_npy_file = Config.Path.Checkpoints.epoch_i_npy_file
        
        epoch_i = 0
        if os.path.exists(epoch_ckpt_dir):
            list_of_files = glob.glob(epoch_i_npy_file)
            epoch_i = np.load(list_of_files[0])
            
            print(f'Latest saved epoch: {epoch_i} ...')
            print(f'Resuming training from epoch {epoch_i+1} ...')

        return epoch_i