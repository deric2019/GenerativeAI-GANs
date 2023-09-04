# File management
import os
import sys

# Data science
import tensorflow as tf

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Classes
from utils.visualization import Visualization

class GenerateSaveImagesCallback(tf.keras.callbacks.Callback):
    '''Generate and save the images during this training epoch'''
    def __init__(self, generator: tf.keras.Model, 
                 dataset_name: str, 
                n_classes: int,  
                latent_dim: int,
                img_during_training_ckpt_dir: str,
                save_every_n_epochs: int):
        """Generate and save the images during this training epoch

        Args:
            generator_f (tf.keras.Model): generator_f model
            dataset_name (str): dataset name
            n_classes (int): number of class labels
            latent_dim (int):  latent dim
            img_during_training_ckpt_dir (str):  path to the folder where the images saves
            save_every_n_epochs (int): save image every n epochs
        """
        super().__init__()
        self.generator = generator
        self.dataset_name = dataset_name
        self.fashion_mnist = True if self.dataset_name == 'fashion_mnist' else False
        self.n_classes = n_classes
        self.latent_dim = latent_dim

        self.img_during_training_ckpt_dir = img_during_training_ckpt_dir

        self.save_every_n_epochs = save_every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        # --------------------------
        # Generate and save images
        # --------------------------
        epoch += 1 
        if epoch % self.save_every_n_epochs == 0:
            # Generate and save the images during this epoch test dataset
            save_path = os.path.join(self.img_during_training_ckpt_dir, f'image_at_epoch_{epoch}.png')
            Visualization.plot_generated_images(gen_o=self.generator, 
                                                n_classes=self.n_classes,
                                                latent_dim=self.latent_dim,
                                                suptitle='', save_path=save_path, 
                                                training=True, fashion_mnist=self.fashion_mnist)