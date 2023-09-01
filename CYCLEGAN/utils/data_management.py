# Data science
import pandas as pd
import tensorflow as tf
import cv2

class DataManagement():
    def take_items_from_batch_dataset(dataset:tf.data.Dataset, n_images: int):
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
    

    def read_video_into_frames(video_path: str):
        '''Read a video and return a list of image tensors
        The images are normalized to [-1,1] with dtype tf.float32

        Args:
            video_path (str): path to the video file
        Returns: a frame list of image tensors and original frame size before resizing
        '''
            # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Create a list to store the resized frames
        frames_list = []

        original_size = None
        # Read, resize, and convert frames to tensors
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Extract the size of the frames
            if not original_size:
                original_size = frame.shape[:2]

            # Convert frame to a TensorFlow tensor
            frame_tensor = tf.convert_to_tensor(frame, tf.float32) 

            # Normalize to [-1,1]
            frame_tensor = (frame_tensor/127.5) - 1
            frames_list.append(frame_tensor)

        # Release the video capture object
        cap.release()

        return frames_list, original_size


    def center_crop_image_square(image, resize=None):
        """
        Center crops the input image to a square of min(height, width)
        Option to resize the cropped image is available
        Args:
            image (tf.Tensor): The input image tensor.
            resize (tuple): resize the cropped image into a new size
        Returns:
            tf.Tensor: The center-cropped image tensor.
        """
        # Extract height and width from size tuple
        image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]
        target_size = tf.minimum(image_height, image_width)

        # Calculate crop dimensions
        crop_top = (image_height - target_size) // 2
        crop_left = (image_width - target_size) // 2

        # Crop the image using tf.image.crop_to_bounding_box
        cropped_image = tf.image.crop_to_bounding_box(image, crop_top, crop_left, target_size, target_size)
        if resize:
            cropped_image = tf.image.resize_with_pad(cropped_image, resize[0], resize[1])

        return cropped_image