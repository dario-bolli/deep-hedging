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

from data_generator import DataGenerator


# Add the parent directory to the search paths to import the libraries.
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, "/".join([dir_path, ".."]))

# Tensorflow settings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.autograph.set_verbosity(0)

if __name__ == '__main__':
    config = yaml.safe_load(open("config.yml"))
    data_params = config['data_params']
    model_params = config['model_params']

    # Simulate the stock price process.
    dataGenerator = DataGenerator(data_params)
    S = dataGenerator.simulate_stock_prices()
    print(S.shape)

    # Assemble the dataset for training and testing.
    # Structure of data:
    #   1) Trade set: [S]
    #   2) Information set: [S]
    #   3) payoff (dim = 1)
    training_dataset = dataGenerator.assemble_data()  # HAVE TO CALL SIMULATE_STOCK_PRICES FIRST, TO CHANGE
    print(type(training_dataset))
    """

    # Compute Black-Scholes prices for benchmarking.
    self.price_BS, self.delta_BS, self.PnL_BS = self.get_Black_Scholes_Prices()

    # Compute the loss value for Black-Scholes PnL
    self.loss_BS = Entropy(
        self.PnL_BS,
        tf.Variable(0.0),
        self.loss_param).numpy()

    # Define model and sub-models
    self.model = self.Define_DH_model()

    # Setup and compile the model
    model = Deep_Hedging_Model(model_params)

    self.submodel = self.Define_DH_Delta_Strategy_Model()
    """
