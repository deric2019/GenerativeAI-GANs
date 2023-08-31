# File management
import os

class Config():

    # -------------------------
    # Parameters for the model
    # -------------------------
    class ModelParam():
        ### Training 
        epochs = 200
        batch_size = 1
        buffer_size = 360

        # Optimizer
        learning_rate = 2e-4
        beta_1 = 0.5
        decay_epoch = epochs/2

    # -------------------------
    # Other settings for during training
    # -------------------------
    class Settings():
        # Save all models or the latest ones
        save_all_models = False
        num_ckpt_to_save = 3
        save_image_every_n_epochs = 1
        n_generated_images_test = 20

    class Name:
        name_A = 'horse'
        name_B = 'zebra'

        name_list_AB = [name_A, name_B]
        name_list_A2B = [f'{name_A}2{name_B}', f'{name_B}2{name_A}']

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
            dataset_name = 'horse2zebra_small'
            dataset_dir = os.path.join(dir, dataset_name)

            base_url = 'http://efrosgans.eecs.berkeley.edu/cyclegan/datasets'
            url = os.path.join(base_url, dataset_name)

            ### Data images zip file
            dataset_compressed = dataset_dir + '.zip'

            ### train, val, test folders
            trainA = os.path.join(dataset_dir, 'trainA')
            trainB = os.path.join(dataset_dir, 'trainB')

            testA = os.path.join(dataset_dir, 'testA')
            testB = os.path.join(dataset_dir, 'testB')

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


        class Results:
            # ------------
            # Results
            # ------------
            results_dir = 'results'

            train_dir = os.path.join(results_dir, 'train')
            image_during_training_gif = os.path.join(train_dir, 'image_during_training.gif')
            loss_png_file = os.path.join(train_dir, 'loss.png')

            test_dir = os.path.join(results_dir, 'test')
            combined_generated_images_png = os.path.join(test_dir, 'combined_generated_images.png')

        class Models:
            dir = 'models'
