from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

BATCH_SIZE = 64
NUM_FEATURES = 8
NUM_CLASSES = 2
SEQ_LEN = 30

from model.cnn_mdl_fn import cnn_model_fn
from model.train import train

if __name__ == '__main__':
    train()
