import tensorflow as tf
from tensorflow.keras import Model

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        inputs = encoder.input
        z_mean, z_log_var, z = encoder(inputs)
        outputs = decoder(z)
        super(VAE, self).__init__(inputs=inputs, outputs=outputs, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        # Trackers for the losses
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
            # Handle both x (single input) or (x, y) if passed via fit()
            if isinstance(data, tuple):
                data = data[0]

            with tf.GradientTape() as tape:
                # 1. Forward Pass
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)
                
                # 2. Reconstruction Loss
                # We use binary_crossentropy, but we need to sum it across 
                # height, width, and channels to get the per-image error.
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.keras.losses.binary_crossentropy(data, reconstruction), 
                        axis=(1, 2)
                    )
                )
                
                # 3. KL Divergence Loss
                # This measures how much our latent distribution deviates from a unit Gaussian.
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                
                # beta < 1.0: Prioritizes reconstruction (sharper images, better identity).
                # beta > 1.0: Prioritizes latent space smoothness (better for interpolation).
                # Since your images are blurry/identical, we drop beta to let reconstruction lead.
                beta = 0.01 
                
                total_loss = reconstruction_loss + (beta * kl_loss)

            # 4. Backward Pass
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            
            # 5. Update Metrics for the logs
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            
            return {m.name: m.result() for m in self.metrics}