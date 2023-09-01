# File management
import os
import sys

# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Data science
import tensorflow as tf

# Classes
from dataloader.fetch_data import FetchData
from dataloader.input_pipeline import InputPipeline
from utils.file_management import FileManagement

class DataLoader:
    def __init__(self, args):
        self.args = args
        self.set_paths()

    def set_paths(self):
        ### Data folder to download zip 
        self.path_data_dir = self.args.data_dir

        ### Data images dir
        self.data_name = self.args.dataset_name
        
        self.path_data_dir = os.path.join(self.path_data_dir, self.data_name)

        self.base_url = self.args.dataset_url
        self.url = os.path.join(self.base_url, self.data_name)

        ### Data images zip file
        self.path_data_compressed = self.path_data_dir + '.zip'

        ### train, val, test folders
        self.path_trainA = os.path.join(self.path_data_dir, 'trainA')
        self.path_trainB = os.path.join(self.path_data_dir, 'trainB')

        self.path_testA = os.path.join(self.path_data_dir, 'testA')
        self.path_testB = os.path.join(self.path_data_dir, 'testB')

    def fetch_data(self):
        '''Download compressed file from URL and extract it'''
        
        # Creating a new data folder if it does not already exists
        FileManagement.create_folder_it_not_already_exists(self.path_data_dir)

        # Download data from url
        FetchData.download_data_from_url(self.url, self.path_data_compressed)

        # Extract compressed file
        FetchData.extract_compressed_file(self.path_data_compressed, self.path_data_dir)

    def load_data_into_dataset(self):
        'Returns a dictionary of '
        print('Loading and preprocessing the dataset ...')
        # Read image paths 
        trainA = tf.data.Dataset.list_files(self.path_trainA + '/*.jpg')
        trainB = tf.data.Dataset.list_files(self.path_trainB + '/*.jpg')

        testA = tf.data.Dataset.list_files(self.path_testA + '/*.jpg')
        testB = tf.data.Dataset.list_files(self.path_testB + '/*.jpg')
        
        # Buffer and batch size for the dataset
        buffer_size = self.args.buffer_size
        batch_size = self.args.batch_size

        # Map image paths to tensor and preprocess pictures
        trainA = trainA.map(
            InputPipeline.load_and_preprocess_image_train, 
            num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=buffer_size).batch(
            batch_size=batch_size, drop_remainder=True)
        
        trainB = trainB.map(
            InputPipeline.load_and_preprocess_image_train, 
            num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=buffer_size).batch(
            batch_size=batch_size, drop_remainder=True)
        
        testA = testA.map(
            InputPipeline.load_and_preprocess_image_test, 
            num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=buffer_size).batch(
            batch_size=batch_size, drop_remainder=True)
        
        testB = testB.map(
            InputPipeline.load_and_preprocess_image_test, 
            num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=buffer_size).batch(
            batch_size=batch_size, drop_remainder=True)       

        print('Finished loading and preprocessing dataset ...')
        print('Returning train_A, train_B, test_A, test_B dataset ...')
        
        return {'trainA': trainA, 'trainB': trainB, 
                'testA' :testA, 'testB': testB}

    ### Main function for data loading
    def load_dataset(self):
        '''Load dataset from data, returns a dict of trainA, trainB, testA, testB'''
        # Download compressed file from URL and extract it 
        self.fetch_data()

        # Load images into a dataset
        # Returns train_A, train_B, test_A, test_B
        return self.load_data_into_dataset()




