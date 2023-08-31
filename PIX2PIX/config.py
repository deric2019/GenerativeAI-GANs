# File management
import os

class Config():
   # -------------------------
    # Parameters for the model
    # -------------------------
    class ModelParam():
        
        img_width = 256
        img_height = 256
        img_channels = 3

        img_size = (img_width, img_height)
        img_shape = (img_width, img_height, img_channels)
        
        ### Training 
        epochs = 200
        batch_size = 4
        buffer_size = 360

        # Optimizer
        learning_rate = 2e-4
        beta_1 = 0.5

    # -------------------------
    # Other settings for during training
    # -------------------------
    class Settings():
        img_type = '.jpg'

        # Save all models or the latest ones
        save_all_models = False
        num_ckpt_to_save = 3
        save_image_every_n_epochs = 1
        n_generated_images_test = 5

        gen_loss_dict = {0:'gen_total_loss',
                           1: 'gen_adv_loss',
                            2: 'gen_l1_loss'}
        loss_to_optimize = gen_loss_dict[0]

    # ----------
    # Paths
    # ----------
    class Path(): 

        class Data:
        # ------------
        # Data 
        # ------------

            ### Data folder to download zip 
            dir = 'data'

            ### Data images dir
            img_name = 'edges2pokemon'
            img_dir = os.path.join(dir, img_name)

            url = 'https://www.kaggle.com/datasets/norod78/sketch2pokemon/download?datasetVersionNumber=1'

            ### Data images zip file
            img_dir_compressed = img_dir + '.zip'

            ### train, val, test folders
            img_train = os.path.join(img_dir, 'train')
            img_val = os.path.join(img_dir, 'val')
            img_test = None

            video = os.path.join(dir, 'squirtle_720p_trimmed.mp4')

        class Checkpoints:
            # ------------
            # Checkpoints
            # ------------
            ### Checkpoints
            dir = 'checkpoints'

            # Epoch
            epoch_ckpt_dir = os.path.join(dir, 'epoch')
            epoch_i_npy_file =  os.path.join(epoch_ckpt_dir, '*.npy')

            # Model: generator and discriminator
            model_ckpt_dir = os.path.join(dir, 'models')
                        
            # Images during training
            image_during_training_names = 'image_at_epoch_*.png'
            img_during_training_ckpt_dir = os.path.join(dir, 'images_during_training')
            img_during_training_images = os.path.join(img_during_training_ckpt_dir, image_during_training_names)

            # csvlog 
            csvlog_ckpt_dir = os.path.join(dir, 'csvlog')
            csvlog_log_file = os.path.join(csvlog_ckpt_dir, 'training.log')

        class Models:
            dir = 'models'

        class Results:
            # ------------
            # Results
            # ------------
            results_dir = 'results'

            train_dir = os.path.join(results_dir, 'train')             
            image_during_training_gif = os.path.join(train_dir, 'image_during_training.gif')
            loss_png_file = os.path.join(train_dir, 'loss.png')

            test_dir = os.path.join('results', 'test')
            combined_generated_images_png = os.path.join(test_dir, 'combined_generated_images.png')



