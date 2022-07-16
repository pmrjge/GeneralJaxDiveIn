import tensorflow as tf
import numpy as np

tf.config.experimental.set_visible_devices([], "GPU")

def load_dataset():
    data = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = data.load_data()

    training_images = training_images / 255.0
    test_images = test_images / 255.0

    train_x = training_images.reshape((-1, 28, 28, 1))
    test = test_images.reshape((-1, 28, 28, 1))

    np.save('./data/fashion_mnist/train_x.npy', train_x, allow_pickle=True)
    np.save('./data/fashion_mnist/train_y.npy', training_labels, allow_pickle=True)
    np.save('./data/fashion_mnist/test_x.npy', test, allow_pickle=True)
    np.save('./data/fashion_mnist/test_y.npy', test_labels, allow_pickle=True)


print('Saving')
load_dataset()
print('Saved.....................')
