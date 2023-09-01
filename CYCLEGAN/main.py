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
            Test.test()

    match args.utils:
        case 'list_devices':
            Utils.list_devices()        

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
    parser.add_argument('--dataset_url', type=str, default='http://efrosgans.eecs.berkeley.edu/cyclegan/datasets',
                        help= 'url link to dataset')
    parser.add_argument('--dataset_name', type=str, required=True, 
                        help= 'name on the dataset to be loaded and trained')
    
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
    # Training loss
    # Training optimizer
    parser.add_argument('--lambda_cycle', type=float, default=10,
                        help= 'cycle loss parameter')
    parser.add_argument('--lambda_identity', type=float, default=0.5,
                        help= 'identity loss parameter')
    
    # Optimizers
    choice_list = ['train', 'test']
    parser.add_argument('-m', '--mode', type=str, required=False, 
                        choices=choice_list , 
                        help=   'train: train the model on the dataset \n' + 
                                'test: generate results from training and on new test images, video')

    choice_list = ['list_devices']
    parser.add_argument('-u','--utils', type=str, required=False, 
                        choices=choice_list, 
                        help= 'list_devices: list physical devices')
    # Parse args
    args = parser.parse_args()
    # Call the main method
    main(args)