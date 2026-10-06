"""Train the GAN on CIFAR-10."""
import numpy as np

from data import load_and_prepare_data
from models import build_discriminator, build_generator, build_gan

MODEL_PATH = 'cifar10_generator.h5'


def train_gan(g_model, d_model, gan_model, dataset, latent_dim,
              epochs=200, batch_size=128):
    batch_per_epoch = int(dataset.shape[0] / batch_size)
    half_batch = int(batch_size / 2)

    for epoch in range(epochs):
        for _ in range(batch_per_epoch):
            # 1. Train Discriminator on REAL images
            idx = np.random.randint(0, dataset.shape[0], half_batch)
            real_images = dataset[idx]
            real_labels = np.ones((half_batch, 1))
            d_loss_real, _ = d_model.train_on_batch(real_images, real_labels)

            # 2. Train Discriminator on FAKE images
            noise = np.random.randn(half_batch, latent_dim)
            fake_images = g_model.predict(noise, verbose=0)
            fake_labels = np.zeros((half_batch, 1))
            d_loss_fake, _ = d_model.train_on_batch(fake_images, fake_labels)

            # 3. Train Generator (via the GAN model) with inverted labels
            gan_noise = np.random.randn(batch_size, latent_dim)
            gan_labels = np.ones((batch_size, 1))
            g_loss = gan_model.train_on_batch(gan_noise, gan_labels)

        print(f"Epoch {epoch+1}/{epochs} | D Loss Real: {d_loss_real:.4f} | "
              f"D Loss Fake: {d_loss_fake:.4f} | G Loss: {g_loss:.4f}")

    # Save the generator for future use
    g_model.save(MODEL_PATH)
    print(f"Training complete. Generator saved as '{MODEL_PATH}'.")


def main():
    latent_dim = 100
    discriminator = build_discriminator()
    generator = build_generator(latent_dim)
    gan = build_gan(generator, discriminator)

    dataset = load_and_prepare_data()
    print(f"Dataset shape: {dataset.shape}")

    train_gan(generator, discriminator, gan, dataset, latent_dim,
              epochs=100, batch_size=128)


if __name__ == '__main__':
    main()
