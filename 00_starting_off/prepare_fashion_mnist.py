import tensorflow as tf
import numpy as np

tf.config.experimental.set_visible_devices([], "GPU")

def load_dataset():
    data = tf.keras.datasets.fashion_mnist
    (train_x, training_labels), (test, test_labels) = data.load_data()


    np.save('./data/fashion_mnist/train_x.npy', train_x)
    np.save('./data/fashion_mnist/train_y.npy', training_labels)
    np.save('./data/fashion_mnist/test_x.npy', test)
    np.save('./data/fashion_mnist/test_y.npy', test_labels)


print('Saving')
load_dataset()
print('Saved.....................')
