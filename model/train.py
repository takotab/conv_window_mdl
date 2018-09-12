import tensorflow as tf
import numpy as np

from model import BATCH_SIZE, NUM_CLASSES, NUM_FEATURES, SEQ_LEN
from model.cnn_mdl_fn import cnn_model_fn


def train():
    # Create the Estimator
    window_classifier = tf.estimator.Estimator(
            model_fn = cnn_model_fn, model_dir = "/tmp/window_classifier_model")

    # replace with https://www.tensorflow.org/guide/datasets#consuming_csv_data
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {"x": np.random.randn(BATCH_SIZE * 1000, NUM_FEATURES * SEQ_LEN)},
            y = np.random.randint(0, NUM_CLASSES, (BATCH_SIZE * 1000, SEQ_LEN)),
            batch_size = BATCH_SIZE,
            num_epochs = None,
            shuffle = True,
            )

    # Train the model
    window_classifier.train(
            input_fn = train_input_fn,
            steps = 20000,
            )
