# Data science
import tensorflow as tf

# --------------------------------
# Help functions to the dataloader
# --------------------------------
class InputPipeline:
    '''Help functions such as loading and pre processing images
    Images are tensors'''

    def load_image(image_file):
        '''Read images and cast them into tf.float32'''
        # Read and decode an image file to a uint8 tensor
        image = tf.io.read_file(image_file)
        image = tf.io.decode_jpeg(image, channels=3)
        image = InputPipeline.resize(image, [256, 256])

        # Convert both images to float32 tensors
        image = tf.cast(image, tf.float32)
        return image
    
    def resize(image, image_size):
        '''Resize an image into wanted shape
        Args:
            image : tensor
            image_size: a tuple or list, one dimension less than tensor 

        Returns:
            _type_: _description_
        '''
        image = tf.image.resize(image, image_size,
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return image
    
    def random_crop(image):
        '''Crop into [256, 256, 3]
        Args:
            image: tensor'''
        cropped_image = tf.image.random_crop(
            image, size=[256, 256, 3])
        return cropped_image
    
    def normalize(image):
        '''  Normalizing the images to [-1, 1]
        Args: 
            image: tensor'''
        image = (image / 127.5) - 1

        return image
    
    def random_jitter(image):
        '''Resizing image to 286x286x3, perform a random crop,
        then flip the image left to right randomly
        Args:
            image: tensor'''
        # resizing to 286 x 286 x 3
        image = InputPipeline.resize(image, [286, 286])

        # randomly cropping to 256 x 256 x 3
        image = InputPipeline.random_crop(image)

        # random mirroring
        image = tf.image.random_flip_left_right(image)

        return image

    def load_and_preprocess_image_train(img_filepath):
        '''Load an image from the train set and preprocess it
        Args:
            img_filepath: path to the image'''
        image = InputPipeline.load_image(img_filepath)
        image = InputPipeline.random_jitter(image)
        image = InputPipeline.normalize(image)
        return image


    def load_and_preprocess_image_test(img_filepath):
        '''Load an image from the train set and preprocess it'''
        image = InputPipeline.load_image(img_filepath)
        image = InputPipeline.normalize(image)
        return image

    

