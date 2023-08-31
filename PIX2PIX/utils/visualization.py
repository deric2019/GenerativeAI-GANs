# File management 
import os
import sys
import glob

# GIF and Images
from PIL import Image
import imageio

# Data science
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Configuration
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Classes
from utils.file_management import FileManagement
from utils.data_management import DataManagement

class Visualization():
    '''Mainly functions that generates images'''
    def plot_generated_images(gen_o: tf.keras.Model, 
                                dataset: tf.data.Dataset,
                                suptitle: str, save_path: str, 
                                block=False, training=True, show_title=True):
        '''Plot a grid of images generated by a generator during training'''
        
        # Plot all the generated images
        n_per_ax = 3
        nrows, ncols = 4, 2 
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols*n_per_ax, figsize=(14,8), constrained_layout = True)
        fig.suptitle(suptitle)

        # Tight layout
        plt.tight_layout(pad=2)
        plt.subplots_adjust(wspace=0.01, hspace=0.01)

        img_tensor_take_ls = DataManagement.take_items_from_batch_dataset(dataset, nrows*ncols)

        title_ls = ['Input image', 'Ground truth', 'Predicted image']
        img_ind = 0
        for i in range(nrows): 
            for j in range(0, ncols*n_per_ax, n_per_ax):
                # Extracting the images
                input_image, target_image = img_tensor_take_ls[img_ind]
                generated_image = gen_o(input_image[tf.newaxis], training=training)
                img_tensor_ls = [input_image, target_image, tf.squeeze(generated_image)] # (256,256,3) [-1,1]
                
                for k, img_tensor in enumerate(img_tensor_ls):
                    # Getting the pixel values in the [0, 1] range to plot.
                    axes[i, j+k].imshow(img_tensor*0.5+0.5)
                    axes[i, j+k].set_axis_off()
                    # Set title only on the first row
                    if i == 0 and show_title:
                        axes[i, j+k].set_title(title_ls[k])

                img_ind += 1

        plt.show(block=block)
        
        if save_path:
            plt.savefig(save_path)
            print(f'Saved and generated image: {save_path} ...')
        
        plt.close()

    def display_image(path_to_image):
        '''Display a single image'''
        img = Image.open(path_to_image)
        img.show()

    def write_gif(path_to_anim_file, path_to_src):
        '''Read images and write them into a GIF file
        Args: 
            path_to_anim_file: path to save the resulting file
            path_to_src: file pattern path for the images to be written to GIF'''

        with imageio.get_writer(path_to_anim_file, mode='I', duration=1) as writer:
            # Find all files matching the name pattern and sort them
            filenames = glob.glob(path_to_src)
            filenames = FileManagement.sort_files_by_number(filenames)
            
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

            image = imageio.imread(filename)
            writer.append_data(image)
        print(f'Saved and generated GIF: {path_to_anim_file} ...')


    def append_images(path_to_appended_file, path_to_src):
        '''Appending images on top of each other
        Args:
            path_to_appended_file: path to save the resulting file
            path_to_src: file pattern path for the images to be written to GIF'''
        
        # Find all files matching the name pattern and sort them
        image_paths = glob.glob(path_to_src)
        image_paths = FileManagement.sort_files_by_number(image_paths)
        
        # Load PNG images
        image_list = [Image.open(image_path) for image_path in image_paths]

        # Make sure all images have the same width
        image_widths = [image.size[0] for image in image_list]
        if len(set(image_widths)) != 1:
            raise ValueError("All images must have the same width for combining.")

        # Calculate the total height of the combined image
        total_height = sum(image.size[1] for image in image_list)

        # Create a new blank image with the same width and total height
        combined_image = Image.new('RGB', (image_widths[0], total_height))

        # Paste each image one below the other
        y_offset = 0
        for image in image_list:
            combined_image.paste(image, (0, y_offset))
            y_offset += image.size[1]

        # Save the combined image to a file
        combined_image.save(path_to_appended_file)

        print(f"Images combined and saved as: {path_to_appended_file}")

    def plot_records(df: pd.DataFrame, save_path: str, 
                     xlabel:str, ylabel: str, title: str, block=False):
        '''Plot from a data frame consisting of epoch, disc_loss and gen_loss
        Args: 
            df: a dataframe with epoch as first column and loss after'''

        fig, ax = plt.subplots(figsize=(8, 6))
        
        # First  column is the epoch, and the rest losses
        for i in range(1, df.shape[1]):
            ax.plot(df.iloc[:, 0], df.iloc[:, i], label=df.columns[i])
        
        xticks = np.linspace(start=1, stop=df.iloc[-1, 0], num=5, dtype=int)
        ax.set_xticks(xticks)
        ax.set_xlabel(xlabel), ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()    
        plt.show(block=block)
        plt.savefig(save_path)
        print(f'Saved and generated plot: {save_path} ...')


if __name__ == '__main__':
    from models.networks import generator
    from dataloader.data_loader import DataLoader

    gen_o = generator.define_unet_generator(model_name='generator')
    dataset_dict = DataLoader.load_dataset()

    Visualization.plot_generated_images(gen_o= gen_o, 
                            dataset= dataset_dict['test'],
                            suptitle= '' ,save_path= '', 
                            block=True, training=False)

