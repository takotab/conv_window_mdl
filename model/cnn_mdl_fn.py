import tensorflow as tf

from model import BATCH_SIZE, NUM_CLASSES, NUM_FEATURES, SEQ_LEN


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, SEQ_LEN, NUM_FEATURES, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
            inputs = input_layer,
            filters = 32,
            kernel_size = [3, 3],
            padding = "same",
            activation = tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [1, 3], strides = 1)

    # Convolutional Layer #2 and Pooling Layer #2 (-1, 30, 7, 64)
    conv2 = tf.layers.conv2d(inputs = pool1,
                             filters = 64,
                             kernel_size = [3, 3],
                             padding = "same",
                             activation = tf.nn.relu)

    pool = tf.layers.max_pooling2d(inputs = conv2, pool_size = [1, 5], strides = 1)

    dropout = tf.layers.dropout(
            inputs = pool, rate = 0.5, training = mode == tf.estimator.ModeKeys.TRAIN)
    # Final Layer
    conv3 = tf.layers.conv2d(inputs = dropout,
                             filters = 2,
                             kernel_size = 1,
                             padding = "same",
                             activation = tf.nn.relu,
                             )
    logits = tf.reshape(conv3, shape = (-1, SEQ_LEN, NUM_CLASSES))
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes"      : tf.argmax(input = logits, axis = 1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name = "softmax_tensor")
        }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.contrib.seq2seq.sequence_loss(logits = tf.cast(logits, tf.float32),
                                            targets = tf.cast(labels, tf.int32),
                                            weights = tf.ones((BATCH_SIZE, SEQ_LEN)),
                                            name = 'loss',
                                            )

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(
                loss = loss,
                global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
                labels = labels, predictions = predictions["classes"])}
    return tf.estimator.EstimatorSpec(
            mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)
