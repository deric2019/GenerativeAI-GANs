# File management
import os
import sys

# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Data science
import tensorflow as tf

# Modules
import models.layers as layers

def define_unet_generator(output_channels=3, norm_type='batchnorm', model_name=None):
    """Modified u-net generator model (https://arxiv.org/abs/1611.07004).

    Args:
        output_channels: Output channels
        norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.

    Returns:
        Generator model
    """
    # Inputs: (bs, 256, 256, 3)
    inputs = layers.input_layer(shape=(256,256,3))

    # Encoder; C64-C128-C256-C512-C512-C512-C512-C512 
    down_filter_list = [64,128,256,512,512,512,512,512]
    down_stack = []
    for i, filters in enumerate(down_filter_list):
        apply_norm = False if i == 0 else True
        down_stack.append(layers.down_sample(filters=filters, kernel_size=(4,4), 
                                            apply_norm=apply_norm))
    
    # Decoder: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
    up_filter_list = [512,512,512,512,256,128,64]
    up_stack = []
    for i, filters in enumerate(up_filter_list):
        apply_dropout = True if i < 3 else False      
        up_stack.append(layers.up_sample(filters=filters, kernel_size=(4,4), 
                                        apply_dropout=apply_dropout))

    last = layers.transpose_conv_layer(
        filters=output_channels, kernel_size=(4,4), strides=(2,2),
        padding='same', activation='tanh')  # (bs, 256, 256, 3)
    
    x = inputs

    # Down sampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Up sampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x =  layers.concatenate_layer()([x, skip])

    # Last layer
    x = last(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)

    if model_name:
        model._name = model_name
    
    return model

def define_patch_discriminator(norm_type='batchnorm', model_name=None):
    """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).

    Args:
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
    target: Bool, indicating whether target image is an input or not.

    Returns:
    Discriminator model
    """
    # Inputs each: (bs,256, 256,3)
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    # Concatenate inputs:  (bs,512, 512,3)
    x = tf.keras.layers.Concatenate()([inp, tar]) # (batch_size, 256, 256, channels*2)

    layer_list = [64,128,256, 512]
    last_index = len(layer_list) - 1
    
    strides = (2,2)
    padding = 'same'
    for i, filters in enumerate(layer_list):
        # No norm on the first layer
        apply_norm = False if i == 0 else True

        # Strides (1,1) and add padding for the second last layer
        if i == last_index:
            x = layers.zero_padding_layer()(x)
            strides = (1,1)
            padding='valid'
                
        x = layers.down_sample(filters=filters, kernel_size=(4,4), 
                                        strides=strides, padding=padding,
                                        apply_norm=apply_norm, norm_type=norm_type)(x)
    # Last layer,     # Patch output
    x = layers.zero_padding_layer()(x)
    patch_out = layers.conv_layer(filters=1, kernel_size=(4,4))(x)

    model = tf.keras.Model(inputs=[inp, tar], outputs=patch_out)

    if model_name:
        model._name = model_name
    return model
    

if __name__ == '__main__':
    # Create generator
    unet_generator=define_unet_generator(model_name='unet_generator')
    patch_discriminator = define_patch_discriminator(model_name='patch_discriminator')

    model_list = [unet_generator, patch_discriminator]

    curr_dir = os.path.dirname(__file__)
    save_dir = 'network_plots'
    save_dir_path = os.path.join(curr_dir, save_dir)

    # Create a dir to store model plots
    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)
    
    for model in model_list:
        # Print model summary
        model.summary()

        # Plot generator and save as png
        model_plot_save_path = os.path.join(save_dir_path, f'{model._name}.png')
        tf.keras.utils.plot_model(model, model_plot_save_path,
                                show_shapes=True, expand_nested=True, dpi=64)
