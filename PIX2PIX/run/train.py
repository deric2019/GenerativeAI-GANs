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
from models.pix2pix import PIX2PIX

from callbacks.epoch import EpochCallback
from callbacks.image import GenerateSaveImagesCallback
from callbacks.model import SaveLoadGeneratorDiscriminatorCallback


class Train:
    '''Class consisting of function acting on the model'''
    def __init__(self, args) -> None:
        self.args = args
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

        # -----------------------------
        # Create instance of cyclegan
        # ------------------------------
        pix2pix_o = PIX2PIX(self.args)

        # -----------------------------
        # Optimizers and loss functions
        # ------------------------------
        # Compile the model
        lr, beta_1 = self.args.learning_rate, self.args.beta_1
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
        epoch_ckpt_callback = EpochCallback(self.path_epoch_ckpt_dir, self.path_epoch_i_npy_file)

        ### Models
        model_ckpt_callback = SaveLoadGeneratorDiscriminatorCallback(model_list=[pix2pix_o.generator, 
                                                                               pix2pix_o.discriminator],
                                                                                model_ckpt_dir=self.path_model_ckpt_dir,
                                                                                num_ckpt_to_save=self.num_ckpt_to_save,
                                                                                save_all=self.save_all_models)
        
        ### Images
        images_callback = GenerateSaveImagesCallback(generator=pix2pix_o.generator, 
                                                     dataset = dataset['val'], 
                                                     img_during_training_ckpt_dir=self.path_img_during_training_ckpt_dir,
                                                     save_every_n_epochs=self.save_image_every_n_epochs
                                                     )
        
        csvlog_ckpt_callback = tf.keras.callbacks.CSVLogger(filename=self.path_csvlog_log_file, append=True)
        
        callback_list = [epoch_ckpt_callback, model_ckpt_callback, images_callback, csvlog_ckpt_callback]

        # -----------------------------
        # Initializing of checkpoints
        # -----------------------------
        # Also returns the latest epoch
        epoch_i = self.initializing_checkpoints_folders()

        # --------------------
        # Train model
        # --------------------
        pix2pix_o.fit(dataset['train'], epochs=self.args.epochs, initial_epoch=epoch_i, callbacks=callback_list)