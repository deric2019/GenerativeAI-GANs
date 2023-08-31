# File management 
import argparse

# Classes
from run.train import Train
from run.test import Test
from run.utils import Utils

def main(args):
    match args.mode:
        case 'train':
            Train.train()
        case 'test':
            Test.test()

    match args.utils:
        case 'list_devices':
            Utils.list_devices()        

if __name__ == '__main__':
    # Object for parsing command line strings into Python objects.
    parser = argparse.ArgumentParser(description='Available argument: mode, utils',
                                    formatter_class=argparse.RawTextHelpFormatter)

    choice_list = ['train', 'test']
    parser.add_argument('-m', '--mode', type=str, required=False, 
                        choices=choice_list , 
                        help=   'train: train the model on the dataset \n' + 
                                'test: generate results from training and on new test images')


    choice_list = ['list_devices']
    parser.add_argument('-u','--utils', type=str, required=False, 
                        choices=choice_list, 
                        help= 'list_devices: list physical devices')
    # Parse args
    args = parser.parse_args()
    
    # Call the main method
    main(args)