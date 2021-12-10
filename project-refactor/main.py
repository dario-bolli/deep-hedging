# Author: Niels Escarfail
# Email: nescarfail@ethz.ch
#
# Last updated: 10/12/2021
#
# Reference: Deep Hedging (2019, Quantitative Finance) by Buehler et al.
# https://www.tandfonline.com/doi/abs/10.1080/14697688.2019.1571683

import sys
import os
import tensorflow as tf
import yaml
from deep_hedging_model import Deep_Hedging_Model
from deep_hedging_model import Delta_SubModel


# Add the parent directory to the search paths to import the libraries.
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, "/".join([dir_path, ".."]))

# Tensorflow settings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.autograph.set_verbosity(0)

if __name__ == '__main__':
    config = yaml.safe_load(open("config.yml"))
    print("HEY")
    print(config)
