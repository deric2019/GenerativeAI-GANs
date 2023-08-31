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
        img_dir_compressed = Config.Path.Data.dataset_compressed
        FetchData.download_data_from_url(url, img_dir_compressed)

        # Extract compressed file
        FetchData.extract_compressed_file(img_dir_compressed, data_dir)

    def load_data_into_dataset():
        print('Loading and preprocessing the dataset ...')
        # Read image paths 
        trainA = tf.data.Dataset.list_files(Config.Path.Data.trainA + '/*.jpg')
        trainB = tf.data.Dataset.list_files(Config.Path.Data.trainB + '/*.jpg')

        testA = tf.data.Dataset.list_files(Config.Path.Data.testA + '/*.jpg')
        testB = tf.data.Dataset.list_files(Config.Path.Data.testB + '/*.jpg')
        
        # Buffer and batch size for the dataset
        buffer_size = Config.ModelParam.buffer_size
        batch_size = Config.ModelParam.batch_size

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
        
        return trainA, trainB, testA, testB

    ### Main function for data loading
    def load_dataset():
        '''Load dataset from data, retunrs train_A, train_B, test_A, test_B'''
        # Download compressed file from URL and extract it 
        DataLoader.fetch_data()

        # Load images into a dataset
        # Returns train_A, train_B, test_A, test_B
        return DataLoader.load_data_into_dataset()

    
if __name__ == '__main__':
    img_train_A, img_train_B, img_test_A, img_test_B = DataLoader.load_dataset()
    print()



