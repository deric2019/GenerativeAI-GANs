# Data science
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 

class VisualizeData:
    def __init__(self, args) -> None:
        self.args = args

    def sample_images(self):
        '''Display some images from the dataset'''

        # Load the MNIST dataset
        (_, _), (digit_images, digit_labels) = tf.keras.datasets.mnist.load_data()

        digit_images = np.expand_dims(digit_images, axis=-1)

        nrows, ncols = 10,10
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10,10))
        fig.suptitle('Ground truth images')
        
        # Looping through each image
        for digit in range(10):
            # Get digit indices and take as many of those as we want
            # Squeeze shape (10,1) into (10,)
            ind = np.squeeze(np.argwhere(digit_labels==digit)[:10])
            
            for j in range(nrows):
                # Reshape (1, 28,28,1) into (28,28,1)
                digit_image = digit_images[ind[j]]
                axes[j, digit].set_axis_off()
                axes[j, digit].imshow(digit_image, cmap='gray') 
                
                # Only plot label on the first row
                if j == 0:
                    axes[j, digit].set_title(str(digit))


        plt.tight_layout()
        plt.show()