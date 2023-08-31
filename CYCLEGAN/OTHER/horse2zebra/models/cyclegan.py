# Data science
import tensorflow as tf

# Modules
import networks as networks

class CYCLEGAN(tf.keras.Model):
    '''Cyclegan model'''
    def __init__(self, lambda_cycle=10, lambda_identity=0.5):
        '''Cyclegan model. Init generators, discriminators, loss trackers and loss parameters
        Args:
            lambda_cycle (float): cycle loss parameter. Defaults to 10
            lambda_identity (float): identity loss parameter. Defaults to 0.5
        '''
        super().__init__()

        # Models
        self.generator_f = networks.define_resnet_generator(model_name='generator_f')
        self.generator_g = networks.define_resnet_generator(model_name='generator_g')

        self.discriminator_x = networks.define_patch_discriminator(model_name='discriminator_x')
        self.discriminator_y = networks.define_patch_discriminator(model_name='discriminator_y')

        # Loss/metric trackers
        self.generator_f_loss_tracker = tf.keras.metrics.Mean(name='generator_f_loss')
        self.generator_g_loss_tracker = tf.keras.metrics.Mean(name='generator_g_loss')
        self.discriminator_x_loss_tracker = tf.keras.metrics.Mean(name='discriminator_x_loss')
        self.discriminator_y_loss_tracker = tf.keras.metrics.Mean(name='discriminator_y_loss')

        # Loss parameter
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(self,
                generator_f_optimizer, 
                generator_g_optimizer ,
                discriminator_x_optimizer,
                discriminator_y_optimizer,
                adv_loss_fn, cycle_loss_fn, identity_loss_fn,
                run_eagerly: bool):
        '''Compiling optimizers and loss functions
        Args:
            generator_f_optimizer (tf.keras.optimizers): optimizer
            generator_g_optimizer (tf.keras.optimizers): optimizer
            discriminator_x_optimizer (tf.keras.optimizers): optimizer
            discriminator_y_optimizer (tf.keras.optimizers): optimizer
            adv_loss_fn (tf.keras.losses): loss
            cycle_loss_fn (tf.keras.losses): loss
            identity_loss_fn (tf.keras.losses): loss
            run_eagerly (bool): true for debugging
        '''
        super().compile()

        # Optimizers
        self.generator_f_optimizer = generator_f_optimizer
        self.generator_g_optimizer = generator_g_optimizer

        self.discriminator_x_optimizer = discriminator_x_optimizer
        self.discriminator_y_optimizer = discriminator_y_optimizer
        
        # Loss functions
        self.adv_loss_fn = adv_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn
        
        self.run_eagerly = run_eagerly

    def train_step(self, data):
        '''Single train step
        Args: 
            data: a tuple of batch dataset, x and y images'''
        # Extract x and y from data
        real_x, real_y = data

        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.
        
            # ---------------       
            # Predictions
            # ---------------
            # y->x, x->y
            fake_x = self.generator_f(real_y, training=True)
            fake_y = self.generator_g(real_x, training=True)

            # x->y->x, y->x->y
            cycled_x = self.generator_f(fake_y, training=True)  
            cycled_y = self.generator_g(fake_x, training=True)

            # Identity mapping
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            ### Discriminator output
            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)          
            
            # ---------------       
            # Compute loss
            # ---------------
            # Generator adversarial loss
            gen_g_adv_loss = self.compute_generator_adversarial_loss(disc_fake_y)
            gen_f_adv_loss = self.compute_generator_adversarial_loss(disc_fake_x)

            # Generator cycle loss
            gen_g_cycle_loss = self.compute_generator_cycle_loss(real_y, cycled_y)
            gen_f_cycle_loss = self.compute_generator_cycle_loss(real_x, cycled_x) 

            gen_g_identity_loss = self.compute_generator_identity_loss(real_y, same_y)
            gen_f_identity_loss = self.compute_generator_identity_loss(real_x, same_x)

            # Total generator loss = adversarial loss + cycle loss + identity loss
            gen_g_total_loss = gen_g_adv_loss + gen_g_cycle_loss + gen_g_identity_loss
            gen_f_total_loss = gen_f_adv_loss + gen_f_cycle_loss + gen_f_identity_loss
            gen_total_loss = gen_g_total_loss + gen_f_total_loss

            # Discriminator loss
            disc_x_loss = self.compute_discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.compute_discriminator_loss(disc_real_y, disc_fake_y)

        # ------------------------------       
        # Compute and apply gradients
        # ------------------------------
        ### Compute the gradients
        generator_g_gradients = tape.gradient(gen_total_loss, 
                                                self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(gen_total_loss, 
                                                self.generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                                    self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                                    self.discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                                    self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                                    self.generator_f.trainable_variables))

        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                        self.discriminator_x.trainable_variables))

        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                        self.discriminator_y.trainable_variables))

        # ---------------       
        # Update metrics
        # ---------------
        self.generator_f_loss_tracker.update_state(gen_f_total_loss)
        self.generator_g_loss_tracker.update_state(gen_g_total_loss)
        self.discriminator_x_loss_tracker.update_state(disc_x_loss)
        self.discriminator_y_loss_tracker.update_state(disc_y_loss)
        
        # Return a dictionary of our loss/metric trackers
        return {
            'gen_f_loss': self.generator_f_loss_tracker.result(),
            'gen_g_loss': self.generator_g_loss_tracker.result(),
            'disc_x_loss': self.discriminator_x_loss_tracker.result(),
            'disc_y_loss': self.discriminator_y_loss_tracker.result(),
        }
    
    # ------------------
    # My own functions
    # ------------------
    def compute_discriminator_loss(self,disc_real_output, disc_generated_output):
        '''Computes the discriminator loss from real and fake discriminator outputs
        Divide the objective by 2 while optimizing D, 
        which slows down the rate at which D learns, relative to the rate of G'''
        real_loss = self.adv_loss_fn(y_true=tf.ones_like(disc_real_output),
                                                            y_pred=disc_real_output)

        generated_loss = self.adv_loss_fn(y_true=tf.zeros_like(disc_generated_output), 
                                                                    y_pred=disc_generated_output)
        
        return (real_loss + generated_loss) * 0.5

    def compute_generator_adversarial_loss(self, disc_generated_output):
        '''Computes the adversarial generator loss from generated discriminator outputs,
        generator outputs and a target'''
        return self.adv_loss_fn(y_true=tf.ones_like(disc_generated_output), 
                                                            y_pred=disc_generated_output)


    def compute_generator_cycle_loss(self, real_image, cycled_image):
        '''Computes the difference between real image and cycled image'''
        return self.lambda_cycle * self.cycle_loss_fn(real_image, cycled_image)
    

    def compute_generator_identity_loss(self, real_image, same_image):
        '''Computes the difference between real image and the same image by the other generator'''
        return self.lambda_cycle * self.lambda_identity * self.identity_loss_fn(real_image, same_image)