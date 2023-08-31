# Data science
import tensorflow as tf

class LearningRateSchedulerCallback(tf.keras.callbacks.Callback):
    '''Learning rate schedular with linearly decay'''
    def __init__(self, initial_lr: int, decay_epoch:int,
                 total_epochs:int , model_optimizer_list: list[tf.keras.optimizers.Adam]):
        '''Initializing parameters
        Args:
            initial_lr (int): initial learning rate
            decay_epoch (int): epoch to start decay learning rate
            total_epochs (int): total number of epochs
            model_optimizer_list (list[tf.keras.optimizers.Adam]): list of optimizers
        '''
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_epoch = decay_epoch
        self.total_epochs = total_epochs
        self.model_optimizer_list = model_optimizer_list

    def on_epoch_begin(self, epoch, logs=None):
        '''Set the learning rates at the beginning
        Args:
            epoch : current epoch number
            logs: Dictionary of metrics. Defaults to None.
        '''
        new_lr = self.linear_decay(epoch)

        # Set new learning rate for each optimizer in the model
        for optimizer in self.model_optimizer_list:
            tf.keras.backend.set_value(optimizer.learning_rate, new_lr)
        print(f'Epoch {epoch+1} - Learning rate: {new_lr}')

    def linear_decay(self, epoch):
        '''Compute linear decay after a certain epoch
        Args:
            epoch: current epoch number 

        Returns:
            new_lr: new learning rate
        '''
        if epoch < self.decay_epoch:
            new_lr = self.initial_lr
        else:
            new_lr = self.initial_lr * (1 - (epoch - self.decay_epoch) / (self.total_epochs-self.decay_epoch))
        return new_lr