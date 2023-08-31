# File management 
import os 
import sys 
from tqdm import tqdm

# Configuration
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import Config

# Classes
from dataloader.data_loader import DataLoader 
from utils.visualization import Visualization
from utils.file_management import FileManagement
from utils.data_management import DataManagement


class Test:
    def test():
        '''Generate images with the latest generator model on the train set and save it to the results results dir'''
        # Create results folders if it does not already exists
        Test.initializing_results_folders()
        Test.train_results()
        Test.test_results()     

    def initializing_results_folders():
        '''Create a new results dir if it does not exists'''
        result_dir = Config.Path.Results.results_dir
        FileManagement.create_folder_it_not_already_exists(result_dir)

        result_train_dir =Config.Path.Results.train_dir
        FileManagement.create_folder_it_not_already_exists(result_train_dir)

        result_test_dir = Config.Path.Results.test_dir
        FileManagement.create_folder_it_not_already_exists(result_test_dir)

        
    def train_results():
        '''Make a GIF of generated images during training and plot loss as function of epochs'''
        # ----------------------------------------
        # Generate GIF from images during training
        # ----------------------------------------
        path_to_anim_file=Config.Path.Results.image_during_training_gif
        path_to_src = Config.Path.Checkpoints.img_during_training_images
        Visualization.write_gif(path_to_anim_file, path_to_src)
        
        # ------------------------
        # Plot losses and metrics
        # ------------------------
        filepath = Config.Path.Checkpoints.csvlog_log_file
        save_path = Config.Path.Results.loss_png_file
        df = DataManagement.read_and_process_training_log(filepath)
        Visualization.plot_records(df[['epoch', 'disc_x_loss','disc_y_loss', 'gen_f_loss', 'gen_g_loss']], 
                     save_path=save_path,
                    xlabel= 'Epoch', ylabel='Loss', title='Training loss', block=False)
        
    def test_results():
        '''Generate many generated outputs from the generators and stack them into a combined file'''
        # Load the test dataset
        _, _, test_A, test_B = DataLoader.load_dataset() 

        # Load the latest generator
        model_ckpt_dir = Config.Path.Checkpoints.model_ckpt_dir
        generator_f = FileManagement.load_trained_generator(model_ckpt_dir, model_name='generator_f') 
        generator_g = FileManagement.load_trained_generator(model_ckpt_dir, model_name='generator_g') 

        # Generate and save final images during training
        result_test_dir = Config.Path.Results.test_dir
        n_generated_images_test = Config.Settings.n_generated_images_test

        for i in tqdm(range(n_generated_images_test)):
            # Show title on the first row only
            show_title = True if i==0 else False
            save_path = os.path.join(result_test_dir, f'generated_image_{i}.png')
            Visualization.plot_generated_images(gen_f=generator_f, gen_g=generator_g,
                                                dataset_A=test_A, dataset_B=test_B,
                                                    suptitle='', save_path=save_path, 
                                                    training=False, show_title=show_title)
            
        # Append all generated images together and save it
        path_to_appended_file = Config.Path.Results.combined_generated_images_png
        file_pattern = 'generated_image_*.png'
        path_to_src = os.path.join(result_test_dir, file_pattern)
        Visualization.append_images(path_to_appended_file, path_to_src)
        
        # Remove images 
        FileManagement.remove_files_in_dir_with_matching_pattern(result_test_dir, file_pattern)
