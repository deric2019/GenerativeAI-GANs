# File management
import os
import sys

# Data science
import tensorflow as tf

# Configuration
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Modules
import models.networks as networks

class PIX2PIX(tf.keras.Model):
    def __init__(self, loss_to_optimize: str):
        """ Init discriminator and generator

        Args:
            discriminator (tf.keras.Model): model
            generator (tf.keras.Model): model
            loss_to_optimize (str): a string to choose which loss to optimize
        """
        super().__init__()

        self.discriminator = networks.define_patch_discriminator(model_name='discriminator')
        self.generator = networks.define_unet_generator(model_name='generator')

        # Loss/metric trackers
        self.disc_loss_metric_tracker = tf.keras.metrics.Mean(name='disc_loss')
        self.gen_total_loss_metric_tracker = tf.keras.metrics.Mean(name='gen_total_loss')
        self.gen_adv_loss_tracker = tf.keras.metrics.Mean(name='gen_adv_loss')
        self.gen_l1_loss_tracker = tf.keras.metrics.Mean(name='gen_l1_loss')

        # Lambda loss parameter
        self.lambda_l1 = 100

        self.loss_to_optimize = loss_to_optimize

    def compile(self,
                generator_optimizer,
                discriminator_optimizer,
                adv_loss_fn,
                l1_loss_fn,
                run_eagerly: bool):
        '''Compiling optimizer, loss function and metrics
        Args:
            generator_optimizer (tf.keras.optimizers): optimizer
            discriminator_optimizer (tf.keras.optimizers): optimizer
            adv_loss_fn (tf.keras.losses): loss
            l1_loss_fn (tf.keras.losses): loss
            run_eagerly (bool): true for debugging
        '''
        super().compile()

        # Optimizers
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        
        # Loss functions
        self.adv_loss_fn = adv_loss_fn
        self.l1_loss_fn = l1_loss_fn

        # Debugging
        self.run_eagerly = run_eagerly

    def train_step(self, data):
        '''Single train step'''
        # --------------------------------------------
        # Split the data into input and target images 
        # --------------------------------------------
        input_images, target_images = data

        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            # Generate images with the generator 
            generated_images = self.generator(input_images, training=True)

            # ------------------------
            # Train the discriminator 
            # ------------------------
            # Discriminator outputs
            disc_real_output = self.discriminator([input_images, target_images], training=True)
            disc_generated_output = self.discriminator([input_images, generated_images], training=True)
            
            # Discriminator adversarial loss
            disc_loss = self.compute_disc_loss(disc_real_output, disc_generated_output)

            # --------------------
            # Train the generator 
            # --------------------            
            # Generator adversarial loss
            total_gen_loss, gan_loss, l1_loss = self.compute_gen_loss(disc_generated_output, 
                                                                       generated_images, target_images)
        # Compute gradients for generator and discriminator
        disc_gradients = disc_tape.gradient(target=disc_loss, sources=self.discriminator.trainable_variables)
        
        # Choose which loss to optimize
        gen_loss_dict = {'gen_total_loss': total_gen_loss,
                            'gen_adv_loss': gan_loss,
                            'gen_l1_loss': l1_loss}
        
        gen_loss = gen_loss_dict[self.loss_to_optimize]
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
        self.gen_total_loss_metric_tracker.update_state(total_gen_loss)
        self.gen_adv_loss_tracker.update_state(gan_loss)
        self.gen_l1_loss_tracker.update_state(l1_loss)
        
        # Return a dictionary of our loss/metric trackers
        return {
            'disc_loss': self.disc_loss_metric_tracker.result(),
            'gen_total_loss': self.gen_total_loss_metric_tracker.result(),
            'gen_adv_loss': self.gen_adv_loss_tracker.result(),
            'gen_l1_loss': self.gen_l1_loss_tracker.result()
        }
    
    # ------------------
    # My own functions
    # ------------------
    def compute_disc_loss(self, disc_real_output, disc_generated_output):
        '''Computes the discriminator loss from real and fake discriminator outputs
        Divide the objective by 2 while optimizing D, 
        which slows down the rate at which D learns, relative to the rate of G'''
                
        real_loss = self.adv_loss_fn(y_true=tf.ones_like(disc_real_output),
                                                            y_pred=disc_real_output)
        fake_loss = self.adv_loss_fn(y_true=tf.zeros_like(disc_generated_output), 
                                                                    y_pred=disc_generated_output)
        return (real_loss + fake_loss) * 0.5

    def compute_gen_loss(self,disc_generated_output, gen_output, target_image):
        '''Computes the generator loss from generated discriminator outputs,
        generator outputs and a target'''
        adv_loss = self.adv_loss_fn(y_true=tf.ones_like(disc_generated_output), 
                                                            y_pred=disc_generated_output)

        # Mean absolute error
        l1_loss = self.l1_loss_fn(target_image, gen_output)

        total_gen_loss = adv_loss + (self.lambda_l1 * l1_loss)

        return total_gen_loss, adv_loss, l1_loss