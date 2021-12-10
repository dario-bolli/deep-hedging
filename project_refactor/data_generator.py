import numpy as np
import QuantLib as ql
import tensorflow as tf
from blackscholes_process import BlackScholesProcess
from sklearn import model_selection

# TODO REMOVE BELOW
# Specify the day (from today) for the delta plot.
delta_plot_day = 15

# European call option (short).
calculation_date = ql.Date.todaysDate()

# Day convention.
day_count = ql.Actual365Fixed()  # Actual/Actual (ISDA)

# Information set (in string)
# Choose from: S, log_S, normalized_log_S (by S0)
information_set = "normalized_log_S"

# Loss function
# loss_type = "CVaR" (Expected Shortfall) -> loss_param = alpha
# loss_type = "Entropy" -> loss_param = lambda
loss_type = "Entropy"

# Other NN parameters
use_batch_norm = False
kernel_initializer = "he_uniform"

activation_dense = "leaky_relu"
activation_output = "sigmoid"
final_period_cost = False

# Reducing learning rate
reduce_lr_param = {"patience": 2, "min_delta": 1e-3, "factor": 0.5}

# Number of bins to plot for the PnL histograms.
num_bins = 30


# TODO REMOVE ABOVE
def train_test_split(data=None, test_size=None):
    """Split simulated data into training and testing sample."""
    xtrain = []
    xtest = []
    for x in data:
        tmp_xtrain, tmp_xtest = model_selection.train_test_split(
            x, test_size=test_size, shuffle=False)
        xtrain += [tmp_xtrain]
        xtest += [tmp_xtest]
    return xtrain, xtest


class DataGenerator:
    def __init__(self, data_params):
        self.seed = data_params['seed']
        self.N = data_params['N']
        self.S0 = data_params['S0']
        self.strike = data_params['strike']
        self.sigma = data_params['sigma']
        self.risk_free = data_params['risk_free']
        self.dividend = data_params['dividend']
        self.Ktrain = data_params['Ktrain']
        self.Ktest_ratio = data_params['Ktest_ratio']
        self.epsilon = data_params['epsilon']
        self.information_set = data_params['information_set']
        self.maturity_date = calculation_date + self.N
        self.payoff_func = lambda x: -np.maximum(x - self.strike, 0.0)

    def simulate_stock_prices(self):
        # Total obs = Training + Testing
        self.nobs = int(self.Ktrain * (1 + self.Ktest_ratio))

        # Length of one time-step (as fraction of a year).
        self.dt = day_count.yearFraction(
            calculation_date, calculation_date + 1)
        self.maturity = self.N * self.dt  # Maturities (in the unit of a year)

        self.stochastic_process = BlackScholesProcess(
            s0=self.S0,
            sigma=self.sigma,
            risk_free=self.risk_free,
            dividend=self.dividend,
            day_count=day_count)

        print("\nRun Monte-Carlo Simulations for the Stock Price Process.\n")
        self.S = self.stochastic_process.gen_path(self.maturity, self.N, self.nobs)
        return self.S

    def assemble_data(self):
        self.payoff_T = self.payoff_func(
            self.S[:, -1])  # Payoff of the call option

        self.trade_set = np.stack((self.S), axis=1)  # Trading set

        if information_set == "S":
            self.infoset = np.stack((self.S), axis=1)  # Information set
        elif information_set == "log_S":
            self.infoset = np.stack((np.log(self.S)), axis=1)
        elif information_set == "normalized_log_S":
            self.infoset = np.stack((np.log(self.S / self.S0)), axis=1)

        # Structure of xtrain:
        #   1) Trade set: [S]
        #   2) Information set: [S]
        #   3) payoff (dim = 1)
        self.x_all = []
        for i in range(self.N + 1):
            self.x_all += [self.trade_set[i, :, None]]
            if i != self.N:
                self.x_all += [self.infoset[i, :, None]]
        self.x_all += [self.payoff_T[:, None]]

        # Split the entire sample into a training sample and a testing sample.
        self.test_size = int(self.Ktrain * self.Ktest_ratio)
        [self.xtrain, self.xtest] = train_test_split(self.x_all, test_size=self.test_size)
        [self.S_train, self.S_test] = train_test_split([self.S], test_size=self.test_size)
        [self.option_payoff_train, self.option_payoff_test] = train_test_split([self.x_all[-1]],
                                                                               test_size=self.test_size)

        # Convert the training sample into tf.Data format (same as xtrain).
        training_dataset = tf.data.Dataset.from_tensor_slices(tuple(self.xtrain))
        return training_dataset.cache()
