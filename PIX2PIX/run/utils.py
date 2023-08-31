# Data science
import tensorflow as tf

class Utils():
    '''Class consisting of additional functions with helpful functions'''
    def list_devices():
        for device in tf.config.list_physical_devices():
            print(device)
