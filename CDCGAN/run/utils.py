import os 
import sys

# Data science
import tensorflow as tf

# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Modules
import models.networks as networks

# Class
from dataloader.visualize_data import VisualizeData

'''Class consisting of additional functions with helpful functions'''
class Utils():
    def __init__(self, args) -> None:
        self.args = args

    def sample_from_data(self):
        vd = VisualizeData(self.args)
        vd.sample_images()
        vd.plot_processed_output()
    
    @staticmethod 
    def list_devices():
        for device in tf.config.list_physical_devices():
            print(device)

    def summary_networks(self): 
        # Create generator
        unet_generator=networks.define_generator(n_classes=self.args.n_classes, latent_dim=self.args.latent_dim, 
                                                 model_name='generator')
        patch_discriminator = networks.define_discriminator(n_classes=self.args.n_classes, model_name='discriminator')

        model_list = [unet_generator, patch_discriminator]

        save_dir_path = 'models/network_plots'

        # Create a dir to store model plots
        if not os.path.exists(save_dir_path):
            os.mkdir(save_dir_path)
        
        for model in model_list:
            # Print model summary
            model.summary()

            # Plot generator and save as png
            model_plot_save_path = os.path.join(save_dir_path, f'{model._name}.png')
            tf.keras.utils.plot_model(model, model_plot_save_path,
                                    show_shapes=True, expand_nested=True, dpi=64)
