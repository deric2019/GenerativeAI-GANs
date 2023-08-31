# Data science
import tensorflow as tf

# --------------------------------
# Help functions to the dataloader
# --------------------------------
class InputPipeline:
    '''Help functions such as loading and precprocessing images
    Images are tensors'''
    def load_image(image_file: str, edges_on_the_left=True):
        '''Split the image (512,512,3) into input and real image (256,256,3)
        Convert them into tf.float32
        Args:
            image_file (str): filepath to the image
            edges_on_the_left (bool): Edges or colored on the left. Default true edges on the left'''
        # Read and decode an image file to a uint8 tensor
        image = tf.io.read_file(image_file)
        image = tf.io.decode_jpeg(image)

        # Split each image tensor into two tensors:
        # - one with a real building facade image
        # - one with an architecture label image 
        # Slice the image in half vertically
        w = tf.shape(image)[1]
        w = w // 2

        input_image = image[:, :w, :] if edges_on_the_left else image[:, w:, :] 
        real_image = image[:, w:, :] if edges_on_the_left else image[:, :w, :] 

        # Convert both images to float32 tensors
        input_image = tf.cast(input_image, tf.float32)
        real_image = tf.cast(real_image, tf.float32)

        return input_image, real_image
    
    
    def resize_image(input_image, real_image, height, width):
        '''Resize image into the wanted shape'''
        input_image = tf.image.resize(input_image, [height, width],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize(real_image, [height, width],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return input_image, real_image
    
    def random_crop(input_image, real_image):
        '''Crop a stacked image'''
        # Stacked image: (512, 256, 3)
        stacked_image = tf.stack([input_image, real_image], axis=0)
        
        # Randomly crop an image into given size 
        cropped_image = tf.image.random_crop(stacked_image, 
                                             size=[2, 256, 256, 3])
        return cropped_image[0], cropped_image[1]
    

    def normalize(input_image, real_image):
        '''  Normalizing the images to [-1, 1]'''
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1

        return input_image, real_image
    
    @tf.function()
    def random_jitter(input_image, real_image):
        '''Resizing to 286x286'''
        input_image, real_image = InputPipeline.resize_image(input_image, real_image, 286, 286)

        # Random cropping back to 256x256
        input_image, real_image = InputPipeline.random_crop(input_image, real_image)

        if tf.random.uniform(()) > 0.5:
            # Random mirroring
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

        return input_image, real_image


    def load_and_preprocess_image_train(image_file):
        '''Load an image from the train set and preprocess it'''
        input_image, real_image = InputPipeline.load_image(image_file)
        input_image, real_image = InputPipeline.random_jitter(input_image, real_image)
        input_image, real_image = InputPipeline.normalize(input_image, real_image)

        return input_image, real_image
    
    def load_and_preprocess_image_val(image_file):
        '''Load an image from the test set and preprocess it'''
        input_image, real_image = InputPipeline.load_image(image_file)
        input_image, real_image = InputPipeline.resize_image(input_image, real_image,
                                       256, 256)
        input_image, real_image = InputPipeline.normalize(input_image, real_image)

        return input_image, real_image
    
    def load_and_preprocess_image_test(image_file):
        '''Load an image from the test set and preprocess it'''
        input_image, real_image = InputPipeline.load_image(image_file)
        input_image, real_image = InputPipeline.resize_image(input_image, real_image,
                                       256, 256)
        input_image, real_image = InputPipeline.normalize(input_image, real_image)

        return input_image, real_image


