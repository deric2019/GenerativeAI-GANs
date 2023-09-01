# File management 
import os 
import sys 
import glob

# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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

class Train:
    '''Class consisting of function acting on the model'''
    def __init__(self, args) -> None:
        self.args = args
        self.name_types = self.args.dataset_name.split('2')
        self.set_paths()
        self.set_diverse_settings()

    def set_paths(self):
        ### Checkpoints
        self.path_ckpt_dir = self.args.checkpoints_dir

        # Epoch
        self.path_epoch_ckpt_dir = os.path.join(self.path_ckpt_dir, 'epoch')
        self.path_epoch_i_npy_file =  os.path.join(self.path_epoch_ckpt_dir, '*.npy')

        # Model: generator and discriminator
        self.path_model_ckpt_dir = os.path.join(self.path_ckpt_dir, 'models')
                    
        # Images during training
        self.path_image_during_training_names = 'image_at_epoch_*.png'
        self.path_img_during_training_ckpt_dir = os.path.join(self.path_ckpt_dir, 'images_during_training')
        self.path_img_during_training_images = os.path.join(self.path_img_during_training_ckpt_dir, self.path_image_during_training_names)

        # csvlog 
        self.path_csvlog_ckpt_dir = os.path.join(self.path_ckpt_dir, 'csvlog')
        self.path_csvlog_log_file = os.path.join(self.path_csvlog_ckpt_dir, 'training.log')


    def set_diverse_settings(self):
        '''Customize how you want'''
        self.save_all_models = False
        self.num_ckpt_to_save = 3
        self.save_image_every_n_epochs = 1


    def initializing_checkpoints_folders(self):
        '''Creating the checkpoint and csv logger dir if it dos not already exists'''
        # Checkpoint dir
        FileManagement.create_folder_it_not_already_exists(self.path_ckpt_dir)

        FileManagement.create_folder_it_not_already_exists(self.path_model_ckpt_dir)
        
        # Create an csvlog dir 
        FileManagement.create_folder_it_not_already_exists(self.path_csvlog_ckpt_dir)

        ### Create an image directory if it does not already exists
        FileManagement.create_folder_it_not_already_exists(self.path_img_during_training_ckpt_dir)

        # Resume epoch number have trained before
        epoch_ckpt_dir = self.path_epoch_ckpt_dir
        epoch_i_npy_file = self.path_epoch_i_npy_file
        
        epoch_i = 0
        if os.path.exists(epoch_ckpt_dir):
            list_of_files = glob.glob(epoch_i_npy_file)
            epoch_i = np.load(list_of_files[0])
            
            print(f'Latest saved epoch: {epoch_i} ...')
            print(f'Resuming training from epoch {epoch_i+1} ...')

        return epoch_i
    

    def train(self):
        '''Train the our model'''
        # --------------
        # Load dataset
        # --------------
        dl = DataLoader(self.args)
        dataset = dl.load_dataset()
        train_AB = tf.data.Dataset.zip((dataset['trainA'], dataset['trainB'])) 
        
        # -----------------------------
        # Create instance of cyclegan
        # ------------------------------
        cyclegan_o = CYCLEGAN(self.args)

        # -----------------------------
        # Optimizers and loss functions
        # ------------------------------
        # Compile the model
        lr, beta_1 = self.args.learning_rate, self.args.beta_1
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
        epoch_ckpt_callback = EpochCallback(self.path_epoch_ckpt_dir, self.path_epoch_i_npy_file)

        ### Learning rate schduler
        model_optimizer_list = [cyclegan_o.generator_f_optimizer, 
                                cyclegan_o.generator_g_optimizer,
                                cyclegan_o.discriminator_x_optimizer,
                                cyclegan_o.discriminator_y_optimizer]
        learning_rate_ckpt_callback = LearningRateSchedulerCallback(initial_lr=lr, 
                                                                    decay_epoch=self.args.epochs/2,
                                                                    total_epochs=self.args.epochs, 
                                                                    model_optimizer_list=model_optimizer_list)
        ### Models, models must have model names
        #  in order to creating folders with its name
        model_ckpt_callback = SaveLoadGeneratorDiscriminatorCallback(model_list=[cyclegan_o.generator_f, 
                                                                                cyclegan_o.generator_g,
                                                                                cyclegan_o.discriminator_x,
                                                                                cyclegan_o.discriminator_y],
                                                                                model_ckpt_dir=self.path_model_ckpt_dir,
                                                                                num_ckpt_to_save=self.num_ckpt_to_save,
                                                                                save_all=self.save_all_models)
        ### Images
        images_callback = GenerateSaveImagesCallback(generator_f=cyclegan_o.generator_f, 
                                                     generator_g=cyclegan_o.generator_g,
                                                     dataset_A = dataset['testA'], 
                                                     dataset_B=dataset['testB'],
                                                     img_during_training_ckpt_dir=self.path_img_during_training_ckpt_dir,
                                                     save_every_n_epochs=self.save_image_every_n_epochs,
                                                     name_types=self.name_types)
        
        csvlog_ckpt_callback = tf.keras.callbacks.CSVLogger(filename=self.path_csvlog_log_file, append=True)
        
        callback_list = [epoch_ckpt_callback, learning_rate_ckpt_callback, model_ckpt_callback, images_callback, csvlog_ckpt_callback]

        # -----------------------------
        # Initializing of checkpoints
        # -----------------------------
        # Also returns the latest epoch
        epoch_i = self.initializing_checkpoints_folders()

        # --------------------
        # Train model
        # --------------------
        cyclegan_o.fit(train_AB, epochs=self.args.epochs, initial_epoch=epoch_i, callbacks=callback_list)



