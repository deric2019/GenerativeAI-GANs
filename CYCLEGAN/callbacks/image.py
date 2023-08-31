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
    def __init__(self, generator_f: tf.keras.Model, 
                 generator_g: tf.keras.Model,
                dataset_A: tf.data.Dataset, 
                dataset_B: tf.data.Dataset,
                img_during_training_ckpt_dir: str,
                save_every_n_epochs: int):
        """Generate and save the images during this training epoch

        Args:
            generator_f (tf.keras.Model): generator_f model
            generator_g (tf.keras.Model): generator_g model
            dataset_A (tf.data.Dataset): dataset_A to feed generator_g
            dataset_B (tf.data.Dataset): dataset_B to feed geneartor_f
            img_during_training_ckpt_dir (str):  path to the folder where the images saves
            save_every_n_epochs (int): save image every n epochs
        """
        super().__init__()
        self.generator_f = generator_f
        self.generator_g = generator_g

        self.dataset_A = dataset_A
        self.dataset_B = dataset_B

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
            Visualization.plot_generated_images(gen_f=self.generator_f, gen_g=self.generator_g,
                                                dataset_A=self.dataset_A, dataset_B=self.dataset_B,
                                                    suptitle='', save_path=save_path, 
                                                    training=True)