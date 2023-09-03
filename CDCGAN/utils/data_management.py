# Data science
import pandas as pd
import tensorflow as tf
import cv2

class DataManagement():
    def take_items_from_batch_dataset(dataset, n_images):
        '''Takes images from batch dataset and return a list of image tensors
        Args:
            dataset (tf.data.Dataset): a batched tf.data.Dataset
            n_images (int): how many images to take '''
        # List of image tensors
        item_take_ls = []
        
        # Compute minimal required number of batches
        batch_size = dataset._batch_size.numpy()
        min_n_required_batches = int(tf.math.ceil(n_images /  batch_size))
        
        # Unbatch the data and store images in list
        dataset_unbatched = tf.data.Dataset.unbatch(dataset.take(min_n_required_batches))
        iterator = iter(dataset_unbatched)
        for _ in range(n_images):
            item = iterator.get_next()
            item_take_ls.append(item)
        return item_take_ls
    

    def read_and_process_training_log(filepath: str):
        '''Return a dataframe of the training log file
        Dropping NA Columns and add one to the epoch column
        Args:
            filepath: path to the csv file to be converted into a dataframe'''
        df = pd.read_csv(filepath, sep=',', header=0)

        # Drop columns with NA
        df = df.dropna(axis=1)

        # Add 1 to the epoch column since it starts from 0
        df['epoch'] = df['epoch'] + 1

        return df

    