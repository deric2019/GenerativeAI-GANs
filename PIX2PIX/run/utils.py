import os 
import sys

# Data science
import tensorflow as tf

# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Class
from dataloader.visualize_data import VisualizeData

'''Class consisting of additional functions with helpful functions'''
class Utils():
    def __init__(self, args) -> None:
        self.args = args

    def sample_from_data(self):
        vd = VisualizeData(self.args)
        vd.sample_images()
        vd.plot_processed_output()
    
    @staticmethod 
    def list_devices():
        
        for device in tf.config.list_physical_devices():
            print(device)
