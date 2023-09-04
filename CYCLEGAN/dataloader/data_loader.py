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
        self.dataset_name = self.args.dataset_name
        
        self.path_dataset_dir = os.path.join(self.path_data_dir, self.dataset_name)

        self.base_url = self.args.dataset_url
        self.url = os.path.join(self.base_url, self.dataset_name)

        ### Data images zip file
        self.path_data_compressed = self.path_dataset_dir + self.args.compressed_type

    def load_dataset(self):
        '''Load dataset from data, returns a dict of trainA, trainB, testA, testB'''
        self.fetch_data()
        self.list_dataset_folders()
        return self.load_data_into_dataset()
    
    def fetch_data(self):
        '''Download compressed file from URL and extract it'''
        
        # Creating a new data folder if it does not already exists
        FileManagement.create_folder_it_not_already_exists(self.path_data_dir)

        # Download data from url
        FetchData.download_data_from_url(self.url, self.path_data_compressed)

        # Extract compressed file
        FetchData.extract_compressed_file(self.path_data_compressed, self.path_data_dir)
    
    
    def list_dataset_folders(self):
        self.dataset_folder_list = [folder for folder in os.listdir(self.path_dataset_dir) if not folder.startswith('.')]
        print(self.dataset_folder_list)
    
    def load_data_into_dataset(self):
        'Returns a dictionary of '
        print('Loading and preprocessing the dataset ...')
        
        # Image type
        img_type = self.args.image_type
        
        # Buffer and batch size for the dataset
        buffer_size = self.args.buffer_size
        batch_size = self.args.batch_size

        # Allocate dataset
        dataset = {}    
        
        for folder in self.dataset_folder_list:
            # Read image paths 
            file_list_path = os.path.join(self.path_dataset_dir, folder + f'/*{img_type}')
            dataset_file_list = tf.data.Dataset.list_files(file_list_path)

            # Map image paths to tensor and preprocess pictures
            if folder.startswith('train'):
                dataset_part = dataset_file_list.map(
                    InputPipeline.load_and_preprocess_image_train, 
                    num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=buffer_size).batch(
                    batch_size=batch_size, drop_remainder=True)
            elif folder.startswith('test'):
                dataset_part = dataset_file_list.map(
                    InputPipeline.load_and_preprocess_image_test, 
                    num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=buffer_size).batch(
                    batch_size=batch_size, drop_remainder=True)        
            dataset[folder] = dataset_part 

        print('Finished loading and preprocessing dataset ...')
        print(f'Returning a dataset with{dataset.keys()}')
        return dataset