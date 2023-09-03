import os
import sys
import tensorflow as tf

# Modules
import models.networks as networks

class CDCGAN(tf.keras.Model):
    def __init__(self, args):
        '''Init discriminator and generator'''
        super().__init__()

        self.args = args
        self.batch_size = self.args.batch_size

        self.discriminator = networks.define_discriminator(self.args.n_classes, model_name='discriminator')
        self.generator = networks.define_generator(self.args.n_classes, self.args.latent_dim, model_name='generator')

        # Loss/metric trackers
        self.disc_loss_metric_tracker = tf.keras.metrics.Mean(name='disc_loss')
        self.gen_loss_metric_tracker = tf.keras.metrics.Mean(name='gen_loss')


    def compile(self,
                generator_optimizer,
                discriminator_optimizer,
                adv_loss_fn,
                run_eagerly: bool):
        '''Compiling optimizer, loss function and metrics
        Args:
            generator_optimizer (tf.keras.optimizers): optimizer
            discriminator_optimizer (tf.keras.optimizers): optimizer
            adv_loss_fn (tf.keras.losses): loss
            run_eagerly (bool): true for debugging
        '''
        super().compile()

        # Optimizers
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        
        # Loss functions
        self.adv_loss_fn = adv_loss_fn

        # Debugging
        self.run_eagerly = run_eagerly
    
    # Override train_step function
    def train_step(self, data):
        '''Single train step'''

        ### Unpack the data.
        real_data_images, real_data_labels = data

        # ------------------------------------------------
        # Generate and fake images for the discriminator
        # ------------------------------------------------
        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(self.args.batch_size, self.args.latent_dim))    

        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            # Using the generator to generate fake images
            fake_images = self.generator([random_latent_vectors, real_data_labels])
            
            # --------------------
            # Train the discriminator 
            # --------------------   
            # The discriminator is then used to classify real and fake images
            disc_real_output = self.discriminator([real_data_images, real_data_labels], training=True)
            disc_generated_output = self.discriminator([fake_images, real_data_labels], training=True)
            
            # Discriminator adversarial loss
            disc_loss = self.compute_disc_loss(disc_real_output, disc_generated_output)
            
            # --------------------
            # Train the generator 
            # --------------------            
            # Generator adversarial loss
            gen_loss = self.compute_gen_loss(disc_generated_output)

        # Compute gradients for generator and discriminator
        disc_gradients = disc_tape.gradient(target=disc_loss, sources=self.discriminator.trainable_variables)
        gen_gradients = gen_tape.gradient(target=gen_loss, 
                                            sources=self.generator.trainable_variables)
        
                # Apply gradients with the optimizer
        self.discriminator_optimizer.apply_gradients(grads_and_vars=zip(disc_gradients, 
                                                                self.discriminator.trainable_variables))
        self.generator_optimizer.apply_gradients(grads_and_vars=zip(gen_gradients, 
                                                                self.generator.trainable_variables))
        # ---------------
        # Update metrics
        # ---------------
        # Compute our own metrics
        self.disc_loss_metric_tracker.update_state(disc_loss)
        self.gen_loss_metric_tracker.update_state(gen_loss)

        return {
            'disc_loss': self.disc_loss_metric_tracker.result(),
            'gen_loss': self.gen_loss_metric_tracker.result()
        }
    

    # ------------------
    # My own functions
    # ------------------
    def compute_disc_loss(self, disc_real_output, disc_generated_output):
        '''Computes the discriminator loss from real and fake discriminator outputs'''
                
        disc_real_loss = self.adv_loss_fn(y_true=tf.ones_like(disc_real_output),
                                                            y_pred=disc_real_output)
        disc_fake_loss = self.adv_loss_fn(y_true=tf.zeros_like(disc_generated_output), 
                                                                    y_pred=disc_generated_output)
        return disc_real_loss + disc_fake_loss


    def compute_gen_loss(self, disc_generated_output):
        '''Computes the generator loss from generated discriminator outputs,
        generator outputs and a target'''
        gen_adv_loss = self.adv_loss_fn(y_true=tf.ones_like(disc_generated_output), 
                                                            y_pred=disc_generated_output)

        return gen_adv_loss