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
    parser.add_argument('--checkpoints_dir', type=str, default=f'checkpoints',
                        help= 'checkpoints dir name')
    parser.add_argument('--results_dir', type=str, default='results',
                        help= 'results dir name')
    
    # Data link and name
    choice_list = ['mnist', 'fashion_mnist']
    parser.add_argument('--dataset_name', type=str, required=False, choices=choice_list,
                        help= 'name on the dataset to be loaded and trained')

    # Training parameters
    parser.add_argument('--latent_dim', type=int, default=100,
                        help= 'latent dimension of sample space')
    parser.add_argument('--n_classes', type=int, default=10,
                        help= 'number of label classes')
    
    parser.add_argument('--epochs', type=int, default=100,
                        help= 'number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
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