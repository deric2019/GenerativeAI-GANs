# File management
import os
import sys

# Data science
import tensorflow as tf

# Configuration
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import Config

# Classes
from dataloader.fetch_data import FetchData
from dataloader.input_pipeline import InputPipeline
from utils.file_management import FileManagement

class DataLoader:
    def fetch_data():
        # Download compressed file from URL and extract it 
        
        # Creating a new data folder if it does not already exists
        data_dir = Config.Path.Data.dir
        FileManagement.create_folder_it_not_already_exists(data_dir)

        # Download data from url
        url = Config.Path.Data.url
        img_dir_compressed = Config.Path.Data.img_dir_compressed
        FetchData.download_data_from_url(url, img_dir_compressed)

        # Extract compressed file
        FetchData.extract_compressed_file(img_dir_compressed, data_dir)

    def load_data_into_dataset():
        print('Loading and preprocessing the dataset ...')
        img_type = Config.Settings.img_type

        img_train_path = Config.Path.Data.img_train
        img_val_path = Config.Path.Data.img_val
        img_test_path = Config.Path.Data.img_test

        # Buffer and batch size for the dataset
        buffer_size = Config.ModelParam.buffer_size
        batch_size = Config.ModelParam.batch_size
        
        img_dataset = {}

        # Read image paths 
        # Map image paths to tensor and preprocess pictures
        img_train = tf.data.Dataset.list_files(img_train_path + f'/*{img_type}')
        img_train = img_train.map(
            InputPipeline.load_and_preprocess_image_train, 
            num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=buffer_size).batch(
            batch_size=batch_size, drop_remainder=True)
        img_dataset['train'] = img_train

        # Img_val if it exists
        if Config.Path.Data.img_val:
            # Read image paths 
            img_val = tf.data.Dataset.list_files(img_val_path + f'/*{img_type}')
            img_val = img_val.map(
                InputPipeline.load_and_preprocess_image_val, 
                num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=buffer_size).batch(
                batch_size=batch_size, drop_remainder=True)
            img_dataset['val'] = img_val
        else:
            img_dataset['val'] = None

        # Img_val if it exists
        if Config.Path.Data.img_test:
            img_test = tf.data.Dataset.list_files(img_test_path + f'/*{img_type}')
            img_test = img_test.map(
                InputPipeline.load_and_preprocess_image_test, 
                num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=buffer_size).batch(
                batch_size=batch_size, drop_remainder=True)
            img_dataset['test'] = img_test
        else:
            img_dataset['test'] = None    

        img_dataset['val'] = img_dataset['test'] if not img_dataset['val'] else img_dataset['val']
        img_dataset['test'] = img_dataset['val'] if not img_dataset['test'] else img_dataset['test']

        print('Finished loading and preprocessing dataset ...')
        print('Returning img_train, img_val, img_test dataset ...')
        
        return img_dataset


    ### Main function for data loading
    def load_dataset():
        '''Load dataset from data, returns a dictionary of train, test, val'''
        # Download compressed file from URL and extract it 
        DataLoader.fetch_data()

        # Load images into a dataset
        # Dictionary of two types
        return DataLoader.load_data_into_dataset()
