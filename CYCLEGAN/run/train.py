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
from models.cyclegan import CYCLEGAN

from callbacks.epoch import EpochCallback
from callbacks.image import GenerateSaveImagesCallback
from callbacks.model import SaveLoadGeneratorDiscriminatorCallback
from callbacks.learning_rate import LearningRateSchedulerCallback

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
        train_A, train_B, test_A, test_B = DataLoader.load_dataset()
        train_AB = tf.data.Dataset.zip((train_A, train_B)) 
        
        # -----------------------------
        # Create instance of cyclegan
        # ------------------------------
        cyclegan_o = CYCLEGAN()

        # -----------------------------
        # Optimizers and loss functions
        # ------------------------------
        # Compile the model
        lr, beta_1 = Config.ModelParam.learning_rate, Config.ModelParam.beta_1
        cyclegan_o.compile(
            generator_f_optimizer= tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1),
            generator_g_optimizer= tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1),
            discriminator_x_optimizer= tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1),
            discriminator_y_optimizer= tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1),
            adv_loss_fn=tf.keras.losses.MeanSquaredError(),
            cycle_loss_fn=tf.keras.losses.MeanAbsoluteError(),
            identity_loss_fn=tf.keras.losses.MeanAbsoluteError(),
            run_eagerly=False
        ) 

        # --------------------
        # Callbacks
        # --------------------
        ### Epoch
        epoch_ckpt_dir = Config.Path.Checkpoints.epoch_ckpt_dir
        epoch_i_npy_file = Config.Path.Checkpoints.epoch_i_npy_file
        epoch_ckpt_callback = EpochCallback(epoch_ckpt_dir, epoch_i_npy_file)

        ### Learning rate schduler
        model_optimizer_list = [cyclegan_o.generator_f_optimizer, 
                                cyclegan_o.generator_g_optimizer,
                                cyclegan_o.discriminator_x_optimizer,
                                cyclegan_o.discriminator_y_optimizer]
        learning_rate_ckpt_callback = LearningRateSchedulerCallback(initial_lr=lr, 
                                                                    decay_epoch=Config.ModelParam.decay_epoch,
                                                                    total_epochs=Config.ModelParam.epochs, 
                                                                    model_optimizer_list=model_optimizer_list)
        ### Models, models must have model names
        #  in order to creating folders with its name
        model_ckpt_dir = Config.Path.Checkpoints.model_ckpt_dir
        num_ckpt_to_save = Config.Settings.num_ckpt_to_save
        save_all_models = Config.Settings.save_all_models
        model_ckpt_callback = SaveLoadGeneratorDiscriminatorCallback(model_list=[cyclegan_o.generator_f, 
                                                                                cyclegan_o.generator_g,
                                                                                cyclegan_o.discriminator_x,
                                                                                cyclegan_o.discriminator_y],
                                                                                model_ckpt_dir=model_ckpt_dir,
                                                                                num_ckpt_to_save=num_ckpt_to_save,
                                                                                save_all=save_all_models)
        ### Images
        img_during_training_ckpt_dir = Config.Path.Checkpoints.img_during_training_ckpt_dir
        save_every_n_epochs = Config.Settings.save_image_every_n_epochs
        images_callback = GenerateSaveImagesCallback(generator_f=cyclegan_o.generator_f, 
                                                     generator_g=cyclegan_o.generator_g,
                                                     dataset_A = test_A, 
                                                     dataset_B=test_B,
                                                     img_during_training_ckpt_dir=img_during_training_ckpt_dir,
                                                     save_every_n_epochs=save_every_n_epochs
                                                     )
        
        csvlog_ckpt_callback = tf.keras.callbacks.CSVLogger(filename=Config.Path.Checkpoints.csvlog_log_file, append=True)
        
        callback_list = [epoch_ckpt_callback, learning_rate_ckpt_callback, model_ckpt_callback, images_callback, csvlog_ckpt_callback]

        # -----------------------------
        # Initializing of checkpoints
        # -----------------------------
        # Also returns the latest epoch
        epoch_i = Train.initializing_checkpoints_folders()

        # --------------------
        # Train model
        # --------------------
        cyclegan_o.fit(train_AB, epochs=Config.ModelParam.epochs, initial_epoch=epoch_i, callbacks=callback_list)


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
