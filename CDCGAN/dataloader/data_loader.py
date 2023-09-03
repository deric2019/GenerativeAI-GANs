import numpy as np
import tensorflow as tf

class DataLoader:
    def __init__(self, args):
        self.args = args

    def load_dataset(self):
        '''Main function to load MNIST images: Preprocess, shuffle and batch'''

        print('Loading the MNIST handwritten dataset ...')
        match self.args.dataset_name:
            case 'mnist':
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            case 'fashion_mnist':
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

        # Merge train and test set
        all_digits = np.concatenate([x_train, x_test])
        all_labels = np.concatenate([y_train, y_test])

        # Scale the pixel values to [0, 1] range, add a channel dimension to
        # the images, and one-hot encode the labels.
        all_digits = all_digits.astype('float32') / 255.0
        all_digits = np.expand_dims(all_digits, axis=-1)
        all_labels =  all_labels.astype('float32')

        # Create tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
        dataset = dataset.shuffle(buffer_size=self.args.buffer_size).batch(self.args.batch_size, drop_remainder=True)
        
        print('Finished loading ...')
        return dataset
