# File management 
import os 
import sys 
from tqdm import tqdm

# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Classes
from dataloader.data_loader import DataLoader 
from utils.visualization import Visualization
from utils.file_management import FileManagement
from utils.data_management import DataManagement


class Test():
    def __init__(self, args) -> None:
        self.args = args
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
        path_to_src = os.path.join(self.path_ckpt_dir, 'images_during_training', 'image_at_epoch_*.png')
        Visualization.write_gif(path_to_anim_file, path_to_src)

        
        # ------------------------
        # Plot losses and metrics
        # ------------------------
        filepath = self.path_csvlog_log_file
        save_path = self.path_loss_png_file
        df = DataManagement.read_and_process_training_log(filepath)
        Visualization.plot_records(df[['epoch', 'disc_loss','gen_total_loss', 'gen_adv_loss', 'gen_l1_loss']], 
                     save_path=save_path, xlabel= 'Epoch', ylabel='Loss', title='Training loss', block=False)
    
    def test_results(self):
        '''Generate many generated outputs from the generators and stack them into a combined file'''
         # ----------------------------------------
        # Generate many images from the test set
        # ----------------------------------------
        # Load the test dataset
        dl = DataLoader(self.args)
        dataset = dl.load_dataset()

        # Load the latest generator
        generator = FileManagement.load_trained_generator(self.path_model_ckpt_dir, model_name='generator') 

        # Generate and save final images during training
        result_test_dir = self.path_test_dir
        n_generated_images_test = self.n_generated_images_test

        for i in tqdm(range(n_generated_images_test)):
            # Show title on the first row only
            show_title = True if i==0 else False
            save_path = os.path.join(result_test_dir, f'generated_image_{i}.png')
            Visualization.plot_generated_images(gen_o=generator, dataset=dataset['test'],
                                                    suptitle='', save_path=save_path, 
                                                    training=False, show_title=show_title)
            
        # Append all generated images together and save it
        path_to_appended_file = self.path_combined_generated_images_png
        file_pattern = 'generated_image_*.png'
        path_to_src = os.path.join(result_test_dir, file_pattern)
        Visualization.append_images(path_to_appended_file, path_to_src)

        # Remove images 
        FileManagement.remove_files_in_dir_with_matching_pattern(result_test_dir, file_pattern)

 