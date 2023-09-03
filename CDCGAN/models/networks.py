# Data science
import os
import tensorflow as tf

def define_generator(n_classes, latent_dim, model_name=None):
    # Generator Inputs (latent vector)
    in_latent = tf.keras.Input(shape=latent_dim)
    x_latent = tf.keras.layers.Dense(units=7*7*128)(in_latent)
    x_latent = tf.keras.layers.ReLU()(x_latent)
    out_latent = tf.keras.layers.Reshape(target_shape=(7,7,128))(x_latent)

    # Label input 
    in_label = tf.keras.Input(shape=(1, ))
    x_label = tf.keras.layers.Embedding(input_dim=n_classes, output_dim=50)(in_label)
    x_label = tf.keras.layers.Dense(units=7*7)(x_label) # Scale up to image dimensions
    out_label = tf.keras.layers.Reshape(target_shape=(7,7,1))(x_label)

    # Concatenate label and latent output
    concat = tf.keras.layers.Concatenate()([out_latent, out_label])

    # Hidden Layer 1: (bs,14,14,128)
    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(4,4), 
                                        strides=(2,2), padding='same')(concat)
    x =  tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Hidden Layer 2: (bs,28,28,128)
    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(4,4), 
                                        strides=(2,2), padding='same')(x)
    x =  tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Hidden Layer 2: (bs,28,28,1)
    outputs = tf.keras.layers.Conv2D(filters=1, kernel_size=(7,7), 
                                    activation='tanh', padding='same')(x)
    
    model = tf.keras.models.Model([in_latent, in_label], outputs)

    if model_name:
        model._name = model_name

    return model        

def define_discriminator(n_classes, use_sigmoid=False, model_name=None):
    # Label input
    in_label = tf.keras.layers.Input(shape=(1, ))
    x_label = tf.keras.layers.Embedding(input_dim=n_classes, output_dim=50)(in_label)
    x_label = tf.keras.layers.Dense(units=28*28)(x_label)
    out_label = tf.keras.layers.Reshape(target_shape=(28,28,1))(x_label)

    # Image input
    in_image = tf.keras.layers.Input(shape=(28,28,1))

    # Concatenate label and image output: (bs,28,28,2)
    concat = tf.keras.layers.Concatenate()([out_label, in_image])

    # Hidden Layer 1: (bs,14,14,64)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), 
                                strides=(2,2), padding='same')(concat)
    x =  tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # Hidden Layer 2: (bs,7,7,128)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), 
                                strides=(2,2), padding='same')(x)
    x =  tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Flatten the shape
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)

    out = tf.keras.layers.Dense(units=1)(x)

    if use_sigmoid:
        out = tf.keras.activations.sigmoid(x)

    model = tf.keras.models.Model([in_image, in_label], out)
    
    if model_name:
        model._name = model_name

    return model


if __name__ == '__main__':
    # Create generator
    generator=define_generator(n_classes=10, latent_dim=100, model_name='generator')
    discriminator = define_discriminator(n_classes=10, model_name='discriminator')

    model_list = [generator, discriminator]

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
