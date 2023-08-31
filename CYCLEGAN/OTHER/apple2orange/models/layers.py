# Data science
import tensorflow as tf
import tensorflow_addons as tfa


### Custom layers
class ReflectionPadding2D(tf.keras.layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super().__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")
    
###########################
### Layer parameters
###########################
class LayerParameters:
    # batch_norm layer parameters
    momentum = 0.9
    epsilon  = 1e-5

    # Dropout layer rate
    rate = 0.5

    # Leaky ReLU alpha
    alpha = 0.2

    # Weight initialization
    kernel_initializer = tf.random_normal_initializer(0., 0.02)

###########################
### Layers
###########################
def input_layer(shape):
    '''Input layer'''
    return tf.keras.layers.Input(shape=shape)

def flatten_layer():
    '''Flatten layer'''
    return tf.keras.layers.Flatten()

def linear_layer(units: int, use_bias=True, kernel_initializer=LayerParameters.kernel_initializer, activation=None):
    '''Linear/Dense layer'''
    return tf.keras.layers.Dense(units = units,\
			kernel_initializer = kernel_initializer,
			use_bias=use_bias, activation=activation)


def conv_layer(filters: int, kernel_size:tuple, 
                strides=(1,1), padding='valid', 
               use_bias=True, kernel_initializer=LayerParameters.kernel_initializer,
               activation=None):
    '''Convolutional layer'''
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                  strides=strides, padding=padding,
                                  use_bias=use_bias, kernel_initializer=kernel_initializer,
                                  activation=activation)

def transpose_conv_layer(filters: int, kernel_size:tuple, 
                         strides=(1,1), padding='valid', 
                         use_bias=True, kernel_initializer=LayerParameters.kernel_initializer, activation=None):
    '''Deconvolutional layer'''
    return tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                                  strides=strides, padding=padding,
                                  use_bias=use_bias, kernel_initializer=kernel_initializer,
                                  activation=activation)

def batch_norm_layer(axis=-1, momentum=LayerParameters.momentum, epsilon=LayerParameters.epsilon):
    '''Batch normalization layer, along the channel dimension
    default momentum=0.9 and epsilon=1e-5 '''
    return tf.keras.layers.BatchNormalization(
             momentum=momentum,
			epsilon=epsilon,
			axis=axis)

def instance_norm_layer(axis=-1, epsilon=LayerParameters.epsilon, 
                        center=False,
                        scale=False,
                        gamma_initializer=LayerParameters.kernel_initializer,
                        beta_initializer='zeros'):
    '''Instance normalization layer'''
    return tfa.layers.InstanceNormalization(axis=axis, 
                                            epsilon=epsilon, 
                                            center=center, 
                                            scale=scale,
                                            beta_initializer=beta_initializer,
                                            gamma_initializer=gamma_initializer)

def drop_out_layer(rate=LayerParameters.rate):
    '''Dropout layer with default rate=0.5'''
    return tf.keras.layers.Dropout(rate=rate)

def relu_layer():
    '''ReLu layer'''
    return tf.keras.layers.ReLU()

def leaky_relu_layer(alpha=LayerParameters.alpha):
    '''Leaky ReLu layer with default alpha=0.2'''
    return tf.keras.layers.LeakyReLU(alpha=alpha)

def concatenate_layer(axis=-1):
    '''Concatenate layer'''
    return tf.keras.layers.Concatenate(axis=axis)

def zero_padding_layer():
    'Zero padding layer'
    return tf.keras.layers.ZeroPadding2D()

###########################
### Layer blocks
###########################
def down_sample(filters:int , kernel_size:tuple, strides=(2,2), padding='same',
                use_bias=False, apply_norm=True, norm_type='instancenorm', apply_relu=True, relu_type='lrelu'):
    '''Downsample an image consisting of Conv2D, 
    default Conv2D => Instancenorm => LeakyRelu'''

    if norm_type == 'instancenorm':
        use_bias = True
    elif norm_type == 'batchnorm':
        use_bias = False

    result = tf.keras.Sequential()

    result.add(
        conv_layer(filters=filters, kernel_size=kernel_size,
                    strides=strides, padding=padding,
                    use_bias=use_bias, kernel_initializer=LayerParameters.kernel_initializer)
    )

    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(batch_norm_layer())
        elif norm_type.lower() == 'instancenorm':
            result.add(instance_norm_layer())    
    
    if apply_relu:
        if relu_type.lower() == 'lrelu':
            result.add(leaky_relu_layer())
        elif relu_type.lower() == 'relu':
            result.add(relu_layer())

    return result

def up_sample(filters: int, kernel_size: tuple, 
              strides=(2,2), padding='same',
               use_bias=False, apply_norm=True, norm_type='instancenorm',
                apply_relu=True, relu_type='relu', apply_dropout=False):
    '''Upsamples an input, 
    default Conv2DTranspose => Instancenorm => Dropout => Relu'''
    if norm_type == 'instancenorm':
        use_bias = True
    elif norm_type == 'batchnorm':
        use_bias = False

    result = tf.keras.Sequential()
    result.add(
        transpose_conv_layer(filters=filters, kernel_size=kernel_size, 
                            strides=strides,padding=padding,
                            use_bias=use_bias, kernel_initializer=LayerParameters.kernel_initializer))

    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(batch_norm_layer())
        elif norm_type.lower() == 'instancenorm':
            result.add(instance_norm_layer())

        if apply_dropout:
            result.add(drop_out_layer(rate=0.5))

    if apply_relu:
        if relu_type.lower() == 'lrelu':
            result.add(leaky_relu_layer())
        elif relu_type.lower() == 'relu':
            result.add(relu_layer())

    return result

def resnet_block(input_res, filters, kernel_size=(3, 3),
                            strides=(1, 1), padding='valid', use_bias=False,
                            apply_norm=True, norm_type='instancenorm',
                            apply_relu= True, relu_type='relu'):
    '''Resnet block consisting of two
    Conv2D => Instance => ReLu
    Uses reflective padding
    Args:
        input_res: input tensor
        filters: number of filters in the conv layers, same for all
        norm_type: either batchnorm or default instancenorm, 
    Returns:
        output: 
    '''
    x = input_res
    for i in range(2):
        # No relu activation after the second conv layer
        if i == 1:
            apply_relu = False
        x = ReflectionPadding2D(padding=(1,1))(x)
        x = down_sample(filters=filters, kernel_size=kernel_size,
                            strides=strides, padding=padding, use_bias=use_bias,
                            apply_norm=apply_norm, norm_type=norm_type, 
                            apply_relu=apply_relu, relu_type=relu_type)(x)
        
    return tf.keras.layers.Add()([input_res, x])
