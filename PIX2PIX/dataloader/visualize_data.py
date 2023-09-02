# File management
import os
import sys

# Configuration
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Data science
import tensorflow as tf
import matplotlib.pyplot as plt 

# Classes
from dataloader.data_loader import DataLoader
from dataloader.input_pipeline import InputPipeline
from utils.data_management import DataManagement

# ------------------------------------------------------
# Mainly visualization functions that plot the dataset
# ------------------------------------------------------
class VisualizeData:
    def __init__(self, args) -> None:
        self.args = args
            
    def sample_images(self):
        '''Reading images from the test folder and plot it'''
        dl = DataLoader(self.args)
        dataset = dl.load_dataset()

        n_per_axis = 2
        nrows, ncols = 4, 3
        fig, axes = plt.subplots(nrows=nrows, ncols=n_per_axis*ncols, figsize=(14,8))
        fig.suptitle('Samples from the test dataset')
        
        # Tight layout
        plt.tight_layout(pad=2.4)
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        
        samples_images = DataManagement.take_items_from_batch_dataset(dataset['train'], nrows*ncols)
        
        img_ind = 0
        for i in range(nrows):
            for j in range(0,ncols*n_per_axis,n_per_axis):
                input_img, real_image = samples_images[img_ind]

                axes[i, j].imshow(input_img*0.5+0.5)
                axes[i, j].set_axis_off()

                axes[i, j+1].imshow(real_image*0.5+0.5)
                axes[i, j+1].set_axis_off()

                if i == 0:
                    axes[i, j].set_title('Input image')
                    axes[i, j+1].set_title('Real image')

                img_ind += 1

        plt.show()

    def plot_processed_output(self):    
        dl = DataLoader(self.args)
        dataset = dl.load_dataset()

        nrows, ncols = 4,4
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14,8))
        plt.suptitle('Preprocessed output')

        # Tight layout
        plt.tight_layout(pad=2.4)
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        
        samples_images = DataManagement.take_items_from_batch_dataset(dataset['test'], nrows*ncols)
        
        for i in range(nrows):
            inp, re = samples_images[i]
            rj_inp, rj_re = InputPipeline.random_jitter(inp, re)

            images = [inp, re, rj_inp, rj_re]
            titles =['Input', 'Real', 'Jittered input', 'Jittered real']
            
            for j, (image, title) in enumerate(zip(images, titles)):
                axes[i,j].imshow(image*0.5+0.5)
                axes[i,j].set_axis_off()
                if i==0:
                    axes[i,j].set_title(title)
    
        plt.show()