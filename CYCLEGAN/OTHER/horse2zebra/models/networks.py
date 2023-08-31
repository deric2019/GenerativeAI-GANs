# File management
import os

# Data science
import tensorflow as tf

# Modules
import layers as layers

def define_resnet_generator(output_channels=3, model_name=None):
    """Resnet generator model (https://arxiv.org/pdf/1703.10593.pdf).

    Args:
        output_channels: Output channels

    Returns:
        Generator model
    """

    ### Input  (bs, 256, 256, 3)
    input_gen = layers.input_layer(shape=(256, 256, 3))
    
    # Padding
    # (bs, 262, 252, 3)
    x = layers.ReflectionPadding2D(padding=(3,3))(input_gen)
    
    ### Encoder: c7s1-64,d128,d256
    filter_layer_list = [64, 128, 256]

    for i, filters in enumerate(filter_layer_list):
        # First layer different setting
        if i == 0:
            kernel_size, strides, padding=(7,7), (1,1), 'valid'
        else:
            kernel_size, strides, padding=(3,3), (2,2), 'same'
        x = layers.down_sample(filters=filters, kernel_size=kernel_size, 
                                        strides=strides, padding=padding,
                                        relu_type='relu')(x)
    
    ### Transformer: R256,R256,R256,R256,R256,R256,R256,R256,R256
    n_res_blocks = 9
    for i in range(n_res_blocks):
        x = layers.resnet_block(x, filters=256)
    
    ### Decoder u128,u64,c7s1-3
    filter_layer_list = [128, 64, output_channels]

    for i, filters in enumerate(filter_layer_list):
        # Last layer different setting and no apply relu
        if i == (len(filter_layer_list) - 1):
            x = layers.ReflectionPadding2D(padding=(3, 3))(x)
            x = layers.conv_layer(filters=filters, kernel_size=(7,7))(x)

        else:
            x = layers.up_sample(filters=filters, kernel_size=(3,3))(x)  
    
    # (bs, 256, 256, 3)
    out_gen = tf.keras.activations.tanh(x)

    model = tf.keras.Model(inputs=input_gen, outputs=out_gen)

    if model_name:
        model._name = model_name
    
    return model

def define_patch_discriminator(use_sigmoid=False, model_name=None):
    """PatchGan discriminator model (https://arxiv.org/pdf/1703.10593.pdf).

    Args:
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
    target: Bool, indicating whether target image is an input or not.

    Returns:
    Discriminator model
    """
    # Inputs each: (bs,256, 256,3)
    inp = layers.input_layer(shape=[256, 256, 3])
    x = inp
    
    # C64-C128-C256-C512
    layer_list = [64,128,256,512]
    last_index = len(layer_list) - 1
    
    strides,padding = (2,2), 'same'

    for i, filters in enumerate(layer_list):
        # No norm on the first layer
        apply_norm = False if i == 0 else True

        # Strides (1,1) and add padding for the second last layer
        if i == last_index:
            x = layers.zero_padding_layer()(x)
            strides, padding= (1,1), 'valid'
                
        x = layers.down_sample(filters=filters, kernel_size=(4,4), 
                                            strides=strides, padding=padding,
                                            apply_norm=apply_norm)(x)
    # Last layer,     # Patch output
    x = layers.zero_padding_layer()(x)
    patch_out = layers.conv_layer(filters=1, kernel_size=(4,4))(x)

    if use_sigmoid:
        patch_out = tf.keras.activations.sigmoid(patch_out)
    model = tf.keras.Model(inputs=inp, outputs=patch_out)

    if model_name:
        model._name = model_name
    return model


if __name__ == '__main__':
    # Create generator
    unet_generator=define_resnet_generator(model_name='resnet_generator')
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
