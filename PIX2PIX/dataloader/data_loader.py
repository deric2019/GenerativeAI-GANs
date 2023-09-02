# File management
import os
import sys

# Data science
import tensorflow as tf

# Configuration
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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

        ### Dataset dir
        self.dataset_name = self.args.dataset_name
        self.path_dataset_dir = os.path.join(self.path_data_dir, self.dataset_name)

        self.base_url = self.args.dataset_url
        self.url = os.path.join(self.base_url, self.dataset_name)

        ### Data images zip file
        self.path_dataset_compressed = self.path_dataset_dir + self.args.compressed_type
        
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
        FetchData.download_data_from_url(self.url, self.path_dataset_compressed)

        # Extract compressed file
        FetchData.extract_compressed_file(self.path_dataset_compressed, self.path_data_dir)


    def list_dataset_folders(self):
<<<<<<< HEAD
        self.dataset_folder_list = [folder for folder in os.listdir(self.path_dataset_dir) if not folder.startswith('.')]
=======
        self.dataset_folder_list = [folder for folder in os.listdir(self.path_dataset_dir) if not f.startswith('.')]
>>>>>>> 110b6bbb56fae00beefd765387c03ea7b278f4e6
        

    def load_data_into_dataset(self):
        print('Loading and preprocessing the dataset ...')
        # Image type
        img_type = self.args.image_type

        # Buffer and batch size for the dataset
        buffer_size = self.args.buffer_size
        batch_size = self.args.batch_size

        # Allocate dataset
        dataset = {}

        # Assign global variables to input pipe line
        input_pipeline = InputPipeline()
        input_pipeline.set_global_variable(self.args.input_on_the_right)

        for folder in self.dataset_folder_list:
            # Read image paths 
            # Map image paths to tensor and preprocess pictures
            file_list_path = os.path.join(self.path_dataset_dir, folder + f'/*{img_type}')
            dataset_file_list = tf.data.Dataset.list_files(file_list_path)
            match folder:
                case 'train': 
                    dataset_part = dataset_file_list.map(
                        input_pipeline.load_and_preprocess_image_train, 
                        num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=buffer_size).batch(
                        batch_size=batch_size, drop_remainder=True)
                case 'val':
                    dataset_part = dataset_file_list.map(
                        input_pipeline.load_and_preprocess_image_val, 
                        num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=buffer_size).batch(
                        batch_size=batch_size, drop_remainder=True)
                case'test':
                    dataset_part = dataset_file_list.map(
                        input_pipeline.load_and_preprocess_image_test, 
                        num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=buffer_size).batch(
                        batch_size=batch_size, drop_remainder=True)
            
            dataset[folder] = dataset_part 

        dataset['val'] = dataset['test'] if 'val' not in dataset else dataset['val']
        dataset['test'] = dataset['val'] if 'test'not in dataset['test'] else dataset['test']

        print('Finished loading and preprocessing dataset ...')
        print('Returning img_train, img_val, img_test dataset ...')
        
        return dataset
