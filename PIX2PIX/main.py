# File management 
import argparse

# Classes
from run.train import Train
from run.test import Test
from run.utils import Utils

def main(args):
    match args.mode:
        case 'train':
            tr = Train(args)
            tr.train()
        case 'test':
            te = Test(args)
            te.test()

    match args.utils:
        case 'list_devices':
            ut = Utils(args)  
            ut.list_devices()
        case 'viz_data':
            ut = Utils(args)  
            ut.sample_from_data()
        case 'summary_networks':
            ut = Utils(args)  
            ut.summary_networks() 


if __name__ == '__main__':
    # Object for parsing command line strings into Python objects.
    parser = argparse.ArgumentParser(description='Available argument: mode, utils',
                                    formatter_class=argparse.RawTextHelpFormatter)
    # Folders to create
    parser.add_argument('--data_dir', type=str, default='data',
                        help= 'data dir name')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints',
                        help= 'checkpoints dir name')
    parser.add_argument('--results_dir', type=str, default='results',
                        help= 'results dir name')
    
    # Data link and name
    choice_list = ['cityscapes', 'edges2handbags', 'edges2shoes', 'facades', 'maps', 'night2day', 'pokemon_pix2pix_dataset']
    parser.add_argument('--dataset_url', type=str, default='http://efrosgans.eecs.berkeley.edu/pix2pix/datasets',
                        help= 'url link to dataset')
    parser.add_argument('--dataset_name', type=str, required=False, choices=choice_list,
                        help= 'name on the dataset to be loaded and trained')
    

    # Dataset settings
    parser.add_argument('--image_type', type=str, default='.jpg',
                        help= 'dataset image type, default jpg but could also be png')
    parser.add_argument('--compressed_type', type=str, default='.tar',
                        help= 'compressed file download type, default tar, can be also be zip')
    parser.add_argument('--input_on_the_right', type=str, default='right',
                        help= 'image pair, input default to the left but could be right')


    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                        help= 'number of epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help= 'batch_size')
    parser.add_argument('--buffer_size', type=int, default=360,
                        help= 'buffer size for shuffle')
    # Training optimizer
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help= 'learning rate for the adam optimizer')
    parser.add_argument('--beta_1', type=float, default=0.5,
                        help= 'beta_1 parameter for the adam optimizer')
    gen_loss_list= ['gen_total_loss', 'gen_adv_loss', 'gen_l1_loss']
    parser.add_argument('--loss_to_optimize', type=str, default='gen_total_loss', choices=gen_loss_list,
                        help= 'choose which loss to optimize, default total generator loss')

    # Training loss
    # Training optimizer
    parser.add_argument('--lambda_l1', type=float, default=100.0,
                        help= 'l1 loss parameter')
    
    # Optimizers
    choice_list = ['train', 'test']
    parser.add_argument('-m', '--mode', type=str, required=False, 
                        choices=choice_list , 
                        help=   'train: train the model on the dataset \n' + 
                                'test: generate results from training and on new test images, video')

    choice_list = ['list_devices', 'viz_data', 'summary_networks']
    parser.add_argument('-u','--utils', type=str, required=False, 
                        choices=choice_list, 
                        help= 'list_devices: list physical devices\n' + 
                        'viz_data: plot samples from dataset\n' + 
                        'summary_networks: plot networks and parameters')
    # Parse args
    args = parser.parse_args()
    
    # Call the main method
    main(args)