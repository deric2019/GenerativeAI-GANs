# File management
import os
import sys

# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Data science
import tensorflow as tf
import matplotlib.pyplot as plt 

# Classes
from dataloader.data_loader import DataLoader
from dataloader.input_pipeline import InputPipeline
from utils.visualization import Visualization
from utils.data_management import DataManagement

# ------------------------------------------------------
# Mainly visualization functions that plot the dataset
# ------------------------------------------------------
class VisualizeData:
    def __init__(self, args) -> None:
        self.args = args

    def sample_images(self):
        '''Reading images from the train folder and plot it, sliced or not'''
        # Load test images which have not been unmodified
        dl = DataLoader(self.args)
        dataset = dl.load_dataset()

        # Figure settings
        nrows, ncols = 2,2 
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
        fig.suptitle('Samples from the data')

        # Image lists
        images_test_A = DataManagement.take_items_from_batch_dataset(dataset['testA'], nrows)
        images_test_B = DataManagement.take_items_from_batch_dataset(dataset['testB'], nrows)

        ab_dict = {0: 'Horse',
                1: 'Zebra'}

        for i, image_A_B in enumerate(zip(images_test_A, images_test_B)):
            for j,image in enumerate(image_A_B):
                axes[i,j].imshow(image*0.5+0.5)
                axes[i,j].set_axis_off()
                if i == 0:
                    axes[i,j].set_title(ab_dict[j])

        plt.show()

    def plot_processed_output(self):
        # Load test images which have not been unmodified
        dl = DataLoader(self.args)
        dataset = dl.load_dataset()

        # Figure settings
        nrows, ncols = 2,2 
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

        # Image lists
        images_test_A = DataManagement.take_items_from_batch_dataset(dataset['testA'], 1)
        images_test_B = DataManagement.take_items_from_batch_dataset(dataset['testB'], 1)

        ab_dict = {0: 'Horse',
                    1: 'Zebra'}

        for i, image_true in enumerate(images_test_A+images_test_B):
            title_base = ab_dict[i]
            axes[i, 0].imshow(image_true*0.5+0.5)
            axes[i,0].set_title(title_base)
            axes[i,0].set_axis_off()

            image_processed = InputPipeline.random_jitter(image_true)
            axes[i, 1].imshow(image_processed*0.5+0.5)
            axes[i,1].set_title(title_base + ' with random jitter')
            axes[i,1].set_axis_off()

        plt.show()



