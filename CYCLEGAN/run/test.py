# File management 
import os 
import sys 
from tqdm import tqdm
import imageio

# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Data science
import numpy as np
import tensorflow as tf

# Classes
from dataloader.data_loader import DataLoader 
from utils.visualization import Visualization
from utils.file_management import FileManagement
from utils.data_management import DataManagement


class Test:
    def __init__(self, args) -> None:
        self.args = args
        self.name_types = self.args.dataset_name.split('2')
        self.set_paths()
        self.set_diverse_settings()

    def set_paths(self):
        # Results
        self.path_results_dir = self.args.results_dir

        self.path_train_dir = os.path.join(self.path_results_dir, 'train')
        self.path_image_during_training_gif = os.path.join(self.path_train_dir, 'image_during_training.gif')
        self.path_loss_png_file = os.path.join(self.path_train_dir, 'loss.png')

        self.path_test_dir = os.path.join(self.path_results_dir, 'test')
        self.path_combined_generated_images_png = os.path.join(self.path_test_dir, 'combined_generated_images.png')

        ### Checkpoints
        self.path_ckpt_dir = self.args.checkpoints_dir

        # Model: generator and discriminator
        self.path_model_ckpt_dir = os.path.join(self.path_ckpt_dir, 'models')
                    
        # Images during training
        self.path_image_during_training_names = 'image_at_epoch_*.png'
        self.path_img_during_training_ckpt_dir = os.path.join(self.path_ckpt_dir, 'images_during_training')
        self.path_img_during_training_images = os.path.join(self.path_img_during_training_ckpt_dir, self.path_image_during_training_names)

        # csvlog 
        self.path_csvlog_ckpt_dir = os.path.join(self.path_ckpt_dir, 'csvlog')
        self.path_csvlog_log_file = os.path.join(self.path_csvlog_ckpt_dir, 'training.log')

        # Videos
        self.path_video_dir = os.path.join(self.args.data_dir, 'video')
        self.path_videoA = os.path.join(self.path_video_dir, self.name_types[0] + '.mp4')
        self.path_videoB = os.path.join(self.path_video_dir, self.name_types[1] + '.mp4')

        self.name_list_A2B = [f'{self.name_types[0]}2{self.name_types[1]}', 
                              f'{self.name_types[1]}2{self.name_types[0]}']

        self.path_video_A2B_gif = os.path.join(self.path_results_dir, f'{self.name_list_A2B[0]}.gif')
        self.path_video_B2A_gif = os.path.join(self.path_results_dir, f'{self.name_list_A2B[0]}.gif')


    def set_diverse_settings(self):
        self.n_generated_images_test = 10


    def test(self):
        '''Generate images with the latest generator model on the train set and save it to the results results dir'''
        # Create results folders if it does not already exists
        self.initializing_results_folders()
        self.train_results()
        self.test_results()     


    def initializing_results_folders(self):
        '''Create a new results dir if it does not exists'''
        FileManagement.create_folder_it_not_already_exists(self.path_results_dir)

        FileManagement.create_folder_it_not_already_exists(self.path_train_dir)

        FileManagement.create_folder_it_not_already_exists(self.path_test_dir)

        
    def train_results(self):
        '''Make a GIF of generated images during training and plot loss as function of epochs'''
        # ----------------------------------------
        # Generate GIF from images during training
        # ----------------------------------------
        path_to_anim_file=self.path_image_during_training_gif
        path_to_src = os.path.join(self.args.checkpoints_dir, 'images_during_training', 'image_at_epoch_*.png')
        Visualization.write_gif(path_to_anim_file, path_to_src)
        
        # ------------------------
        # Plot losses and metrics
        # ------------------------
        filepath = self.path_csvlog_log_file
        save_path = self.path_loss_png_file
        df = DataManagement.read_and_process_training_log(filepath)
        Visualization.plot_records(df[['epoch', 'disc_x_loss','disc_y_loss', 'gen_f_loss', 'gen_g_loss']], 
                     save_path=save_path,
                    xlabel= 'Epoch', ylabel='Loss', title='Training loss', block=False)
        

    def test_results(self):
        '''Generate many generated outputs from the generators and stack them into a combined file'''
        # Load the test dataset
        dl = DataLoader(self.args)
        dataset = dl.load_dataset()

        # Load the latest generator
        model_ckpt_dir = self.path_model_ckpt_dir
        generator_f = FileManagement.load_trained_generator(model_ckpt_dir, model_name='generator_f') 
        generator_g = FileManagement.load_trained_generator(model_ckpt_dir, model_name='generator_g') 

        # Generate and save final images during training
        result_test_dir = self.path_test_dir
        n_generated_images_test = self.n_generated_images_test

        for i in tqdm(range(n_generated_images_test)):
            # Show title on the first row only
            show_title = True if i==0 else False
            save_path = os.path.join(result_test_dir, f'generated_image_{i}.png')
            Visualization.plot_generated_images(gen_f=generator_f, gen_g=generator_g,
                                                dataset_A=dataset['testA'], dataset_B=dataset['testB'],
                                                    name_types=self.name_types, save_path=save_path, 
                                                    training=False, show_title=show_title)
            
        # Append all generated images together and save it
        path_to_appended_file = self.path_combined_generated_images_png
        file_pattern = 'generated_image_*.png'
        path_to_src = os.path.join(result_test_dir, file_pattern)
        Visualization.append_images(path_to_appended_file, path_to_src)
        
        # Remove images 
        FileManagement.remove_files_in_dir_with_matching_pattern(result_test_dir, file_pattern)


    def video_results(self):
        '''Transform a video into the other style, save as GIFs'''
        # Paths to the videos to be transformed
        video_paths = [self.path_videoA, self.path_videoB]
        model_names = ['generator_g', 'generator_f']
        save_paths = [self.path_video_A2B_gif, self.path_video_B2A_gif]
        process_list = self.name_list_A2B
        
        # Check if videos exists
        for video_path, model_name, save_path, process in zip(video_paths, model_names, save_paths, process_list):
            # If video file exists
            if os.path.exists(video_path):
                FileManagement.create_folder_it_not_already_exists(self.path_video_dir)

                # Load the latest generator
                generator = FileManagement.load_trained_generator(self.path_model_ckpt_dir, model_name) 

                def create_frame_list(video_path: str, generator_compatible_image_size: tuple, 
                                      customized_center_crop_square_size: tuple, apply_center_crop_squared=True):
                    """Returns the and the generated frame list of a video
                    Args:
                        video_path (str): path to the video
                        generator_compatible_image_size (tuple): 2D tuple of an image size = (width, height) compatible with the generator
                        customized_center_crop_square_size (tuple): customize the center crop size, else just center crop to the min of (height, width)
                        apply_center_crop_squared: center crop or not before resizing to generator_compatible_image_size size
                    Returns:
                        a zip of real and generated frames, and videos original image size (height, width)
                    """

                    original_frame_list, original_size = DataManagement.read_video_into_frames(video_path)
                    new_frame_list = []
                    generated_frame_list = []

                    crop_size = original_size

                    print(f'Generating fake frames from {process}...')
                    for frame in tqdm(original_frame_list): 
                        # Center crop image if true
                        if apply_center_crop_squared:
                            # If we want to center crop int a specif size or just the min of (height, width)
                            resize = customized_center_crop_square_size if customized_center_crop_square_size else None 
                            frame = DataManagement.center_crop_image_square(frame, resize)
                            # Store crop size
                            if crop_size:
                                crop_size = (tf.shape(frame)[0],  tf.shape(frame)[1])
                        
                        # Append the frame to the new frame list
                        new_frame_list.append(frame)

                        # Resize the frame into a size compatible with the generator
                        frame = tf.image.resize(frame, generator_compatible_image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

                        # Insert new axis so we can feed into the generator, 4D tensors
                        generated_frame = generator(frame[tf.newaxis])
                        
                        # Resize it back to crop_size, which can be cropped or original
                        generated_frame = tf.image.resize(generated_frame, crop_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

                        # Append 3D tensors
                        generated_frame_list.append(tf.squeeze(generated_frame))
                    
                    # Return a list of tuples with real, and fake frames
                    return zip(new_frame_list, generated_frame_list)
                
                # Create real and generated frame list for the videos A and B
                generator_compatible_image_size = (256, 256)
                frame_pair_list = create_frame_list(video_path, generator_compatible_image_size, 
                                                    customized_center_crop_square_size=generator_compatible_image_size, apply_center_crop_squared=True)
                
                # Concatenate real and generated images
                # Convert to np.unint8 in the range [0,255] so that we can write GIF with imageio.mimsave
                frame_pair_list = [((tf.concat(frame_pair, axis=0).numpy() + 1)*127.5).astype(np.uint8) for frame_pair in frame_pair_list]
                
                # Write frame lists into GIFs
                print(f'Rendering {process} frames to GIF ...')
                imageio.mimsave(save_path, frame_pair_list, format='GIF', duration=1e-4)