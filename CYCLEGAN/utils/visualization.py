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
    def plot_generated_images(gen_f: tf.keras.Model, gen_g:tf.keras.Model,
                            dataset_A: tf.data.Dataset, dataset_B: tf.data.Dataset,
                            name_types=['', ''] ,suptitle=None, save_path= None, 
                            block=False, training=True, show_title=True):
        """Plot generated images

        Args:
            gen_f (tf.keras.Model): generator f
            gen_g (tf.keras.Model): generator g
            dataset_A (tf.data.Dataset): dataset A
            dataset_B (tf.data.Dataset): dataset B
            name_types (_type_, optional): list of names of the two types. Defaults to None
            suptitle (_type_, optional): if subtitle. Defaults to None.
            save_path (_type_, optional): if save path. Defaults to None.
            block (bool, optional): if plt block. Defaults to False.
            training (bool, optional): generator training mode. Defaults to True.
            show_title (bool, optional): if show title. Defaults to True.
        """

        # Each column/ncols represent a category and row/nrows samples of that category
        nrows, ncols = 4, 2
        n_per_axis = 3
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols*n_per_axis, figsize=(14,8))

        # Tight layout
        plt.tight_layout(pad=2.4)
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        if suptitle:
            plt.suptitle(suptitle)

        # Unbatch dataset and store image tensors in list
        real_x = DataManagement.take_items_from_batch_dataset(dataset_A, nrows)
        real_y = DataManagement.take_items_from_batch_dataset(dataset_B, nrows)

        # Looping through each row
        for i, (input_x, input_y) in enumerate(zip(real_x, real_y)):
            # Generate fake images, Insert a fourth axis to input since the generator requires it
            fake_x = gen_f(input_y[tf.newaxis])
            fake_y = gen_g(input_x[tf.newaxis])

            # Generate cycled images
            cycle_x = gen_f(fake_y)
            cycle_y = gen_g(fake_x)

            # Create a list to iterate over
            ls = [(input_x, fake_y, cycle_x), 
                    (input_y, fake_x, cycle_y)]
    
            for j, (input, fake, cycle) in zip(range(0, ncols*n_per_axis, n_per_axis), ls):
                axes[i, j].imshow(input*0.5+0.5)
                axes[i, j].set_axis_off()

                axes[i, j+1].imshow(tf.squeeze(fake)*0.5+0.5)
                axes[i, j+1].set_axis_off()

                axes[i, j+2].imshow(tf.squeeze(cycle)*0.5+0.5)
                axes[i, j+2].set_axis_off()

                if i == 0 and show_title:
                    name_types_temp = name_types if j % 2 == 0 else name_types[::-1]

                    axes[i, j].set_title('Real ' + name_types_temp[0])
                    axes[i, j+1].set_title('Fake ' + name_types_temp[1])
                    axes[i, j+2].set_title('Cycle ' + name_types_temp[0])

        plt.show(block=block)

        # Save image to path
        if save_path:
            plt.savefig(save_path)
            print(f'Saved and generated image: {save_path} ...')

        plt.close(fig=fig)

    def display_image(path_to_image):
        '''Display a single image'''
        img = Image.open(path_to_image)
        img.show()

    def write_gif(path_to_anim_file, path_to_src, duration=1):
        '''Read images and write them into a GIF file
        Args: 
            path_to_anim_file: path to save the resulting file
            path_to_src: file pattern path for the images to be written to GIF
            duration: how long an image shows during GIF'''

        with imageio.get_writer(path_to_anim_file, mode='I', duration=duration) as writer:
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
        
