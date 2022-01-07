#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 15:13:08 2021

@author: gneven
"""

import sys, os

sys.path.insert(0, os.getcwd() + "/deep-hedging")

# from IPython.display import clear_output

import numpy as np
import QuantLib as ql
import tensorflow as tf
from scipy.stats import norm

import pandas as pd

from tqdm import tqdm

import warnings
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, \
    ReduceLROnPlateau
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

from stochastic_processes import BlackScholesProcess
from instruments import EuropeanCall
from deep_hedging import *  # Deep_Hedging_Model_LSTM, Deep_Hedging_Model_Transformer, Delta_SubModel

from loss_metrics import Entropy, CVaR
from utilities import train_test_split

import argparse

# print("\nFinish installing and importing all necessary libraries!")

if __name__ == '__main__':
    function_mappings = {
        'Deep_Hedging_Model_LSTM': Deep_Hedging_Model_LSTM,
        'Deep_Hedging_Model_Transformer': Deep_Hedging_Model_Transformer,
        'Deep_Hedging_Model_LSTM_CLAMP': Deep_Hedging_Model_LSTM_CLAMP,
        'Deep_Hedging_Model_MLP_CLAMP': Deep_Hedging_Model_MLP_CLAMP,
        'Deep_Hedging_Model_TCN_CLAMP': Deep_Hedging_Model_TCN_CLAMP,
        'Deep_Hedging_Model_TCN': Deep_Hedging_Model_TCN,
        'Deep_Hedging_Model_MLP': Deep_Hedging_Model
    }
    lis = ""
    for s in function_mappings.keys():
        lis += s + ", "

    parser = argparse.ArgumentParser(description=("Run Deep Hedging scripts, use -h to list available option. \n To "
                                                  "select a specific model, use the command --model then the function "
                                                  "name, note that the function name should be in deephedging .init \n"
                                                  "the script will automatically use available GPU, if multiple, "
                                                  "the Mirror strategy is used, best practice is 1 GPU"))

    parser.add_argument('--N', default=30, type=int,
                        help='Number of time steps (in days) default : 30')

    parser.add_argument('--S0', default=100.0, type=np.double,
                        help='Stock price at time = 0 default : 100.0')

    parser.add_argument('-Ktrain', default=1 * (10 ** 5), type=int,
                        help='Size of training sample default : 10^5')

    parser.add_argument('--epsilon', default=0.0, type=np.double,
                        help='Transaction cost default : 0.0')

    parser.add_argument('--loss', default="Entropy", type=str,
                        help='Loss function : \n \
                        loss_type = "CVaR" (Expected Shortfall) -> loss_param = alpha \n \
                        loss_type = "Entropy" -> loss_param = lambda : 0.0')

    parser.add_argument('--info_set', default="normalized_log_S", type=str,
                        help='Information set (in string) \n \
                            Choose from: S, log_S, normalized_log_S (by S0)')

    parser.add_argument('--d', default=1, type=int,
                        help='Number of hidden layers default: 1')
    parser.add_argument('--m', default=15, type=int,
                        help='Number of neurons in each hidden layer default: 15')
    parser.add_argument('--maxT', default=5, type=int,
                        help='Time step default: 5')

    parser.add_argument('--batch', default=256, type=int,
                        help='batch_size default: 256')

    parser.add_argument('--lr', default=0.001, type=np.double,
                        help='learning rate default: 1e-2')

    parser.add_argument('--epochs', default=50, type=int,
                        help='Number of epochs default: 50')

    parser.add_argument('--model', dest='input_model',
                        default="Deep_Hedging_Model_LSTM",
                        help='Model you want to use among %s, default : Deep_Hedging_Model_LSTM' % lis[:-2])

    parser.add_argument('--figname', default="Default", type=str,
                        help='Name for output (fig and file)  default : combination of arguments (m, d, maxT, '
                             'epsilon, args.input_model)')

    parser.add_argument('--outdir', default="Default", type=str,
                        help='Name for output dir default : working directory')

    # parser.print_help()
    args = parser.parse_args()
    # Geometric Brownian Motion.
    N = args.N  # Number of time steps (in days)

    S0 = args.S0  # Stock price at time = 0
    sigma = 0.2  # Implied volatility
    risk_free = 0.0  # Risk-free rate
    dividend = 0.0  # Continuous dividend yield

    Ktrain = args.Ktrain  # Size of training sample.
    Ktest_ratio = 0.2  # Fraction of training sample as testing sample.
    initial_wealth = 0.0
    # European call option (short).
    strike = S0
    payoff_func = lambda x: -np.maximum(x - strike, 0.0)
    calculation_date = ql.Date.todaysDate()
    maturity_date = ql.Date.todaysDate() + N

    # Day convention.
    day_count = ql.Actual365Fixed()  # Actual/Actual (ISDA)

    # Proportional transaction cost.
    epsilon = args.epsilon

    # Information set (in string)
    # Choose from: S, log_S, normalized_log_S (by S0)
    if args.info_set in ['S', 'log_S', 'normalized_log_S']:

        information_set = args.info_set
    else:
        warnings.warn(
            "invalid info_set inputs, must be one of S, log_S, normalized_log_S, your input: %s, set to default" % args.info_set)
        information_set = "normalized_log_S"

        # Loss function
    # loss_type = "CVaR" (Expected Shortfall) -> loss_param = alpha 
    # loss_type = "Entropy" -> loss_param = lambda 

    loss_type = args.loss
    if args.loss in ['Entropy', 'CVaR']:

        loss_type = args.loss
    else:
        warnings.warn("invalid info_set inputs, must be one of CVaR,Entropy your input: %s,set to default" % args.loss)
        information_set = 'Entropy'

    loss_param = 1.0

    lr = args.lr  # Learning rate

    # Neural network (NN) structure
    m = args.m  # Number of neurons in each hidden layer.
    d = args.d  # Number of hidden layers (Note including input nor output layer)
    maxT = args.maxT

    # Neural network training parameters
    batch_size = args.batch  # Batch size
    epochs = args.epochs  # Number of epochs

    # Other parameters
    use_batch_norm = True
    kernel_initializer = "he_uniform"

    activation_dense = "leaky_relu"
    activation_output = "leaky_relu"
    final_period_cost = False

    delta_constraint = (0.0, 1.0)
    share_strategy_across_time = False
    cost_structure = "constant"

    # Other control flags for development purpose.
    mc_simulator = "Numpy"  # "QuantLib" or "Numpy"
    seed = 0  # Random seed. Change to have deterministic outcome.

    # Total obs = Training + Testing
    nobs = int(Ktrain * (1 + Ktest_ratio))

    # Length of one time-step (as fraction of a year).
    dt = day_count.yearFraction(calculation_date, calculation_date + 1)
    maturity = N * dt  # Maturities (in the unit of a year)

    # S0 is init stock price, sigma = Volatility, risk_free = ?, ?,time Day convention
    stochastic_process = BlackScholesProcess(s0=S0, sigma=sigma, risk_free=risk_free, \
                                             dividend=dividend, day_count=day_count, seed=seed)

    S = stochastic_process.gen_path(maturity, N, nobs)

    print("\n\ns0 = " + str(S0))
    print("sigma = " + str(sigma))
    print("risk_free = " + str(risk_free) + "\n")
    print("Number of time steps = " + str(N))
    print("Length of each time step = " + "1/365\n")
    print("Simulation Done!")

    # @title <font color='Blue'>**Prepare data to be fed into the deep hedging algorithm.**</font>

    payoff_T = payoff_func(S[:, -1])  # Payoff of the call option

    trade_set = np.stack((S), axis=1)  # Trading set
    print(information_set)
    if information_set == "S":
        info = np.stack((S), axis=1)  # Information set
    elif information_set == "log_S":
        info = np.stack((np.log(S)), axis=1)
    elif information_set == "normalized_log_S":
        info = np.stack((np.log(S / S0)), axis=1)
    else:
        raise Exception("There is a bug in my code, invalid information_set, yet it should have been taken care of by "
                        "the parser")
    call = EuropeanCall()
    delta_BS = np.transpose(call.get_BS_delta(S=np.transpose(trade_set), sigma=sigma,
                                              risk_free=risk_free, dividend=dividend, K=strike,
                                              exercise_date=maturity_date,
                                              calculation_date=calculation_date,
                                              day_count=day_count, dt=dt))
    # Structure of xtrain:
    #   1) Trade set: [S]
    #   2) Information set: [S]
    #   3) payoff (dim = 1)
    x_all = []
    for i in range(N + 1):
        x_all += [trade_set[i, :, None]]
        if "CLAMP" in args.input_model:
            x_all += [delta_BS[i, :, None]]

        if i != N:
            x_all += [info[i, :, None]]
    x_all += [payoff_T[:, None]]

    # Split the entire sample into a training sample and a testing sample.
    test_size = int(Ktrain * Ktest_ratio)
    [xtrain, xtest] = train_test_split(x_all, test_size=test_size)
    [S_train, S_test] = train_test_split([S], test_size=test_size)
    [option_payoff_train, option_payoff_test] = \
        train_test_split([x_all[-1]], test_size=test_size)

    print("Finish preparing data!")

    # @title <font color='Blue'>**Run the Deep Hedging Algorithm (Recurrent Network)!**</font>
    optimizer = Adam(learning_rate=lr)

    # Setup and compile the model
    gpus = tf.config.list_logical_devices('GPU')
    N_GPU = len(gpus)
    print("Num GPUs Available: ", N_GPU)

    if N_GPU == 0:
        model_recurrent = function_mappings[args.input_model](N=N, d=d, m=m, risk_free=risk_free,
                                                              dt=dt, strategy_type="recurrent", epsilon=epsilon,
                                                              maxT=maxT,
                                                              use_batch_norm=use_batch_norm,
                                                              kernel_initializer=kernel_initializer,
                                                              activation_dense=activation_dense,
                                                              activation_output=activation_output,
                                                              final_period_cost=final_period_cost,
                                                              delta_constraint=delta_constraint
                                                              )
    elif N_GPU == 1:
        with tf.device(gpus[0].name):
            model_recurrent = function_mappings[args.input_model](N=N, d=d, m=m, risk_free=risk_free,
                                                                  dt=dt, strategy_type="recurrent", epsilon=epsilon,
                                                                  maxT=maxT,
                                                                  use_batch_norm=use_batch_norm,
                                                                  kernel_initializer=kernel_initializer,
                                                                  activation_dense=activation_dense,
                                                                  activation_output=activation_output,
                                                                  final_period_cost=final_period_cost,
                                                                  delta_constraint=delta_constraint)

    elif N_GPU > 1:
        print("Use all available GPU using Mirrored Strategy")
        strategy = tf.distribute.MirroredStrategy(gpus)
        with strategy.scope():
            model_recurrent = function_mappings[args.input_model](N=N, d=d, m=m, risk_free=risk_free,
                                                                  dt=dt, strategy_type="recurrent", epsilon=epsilon,
                                                                  maxT=maxT,
                                                                  use_batch_norm=use_batch_norm,
                                                                  kernel_initializer=kernel_initializer,
                                                                  activation_dense=activation_dense,
                                                                  activation_output=activation_output,
                                                                  final_period_cost=final_period_cost,
                                                                  delta_constraint=delta_constraint)

    else:
        warnings.warn("Strange number of GPU")

    loss = Entropy(model_recurrent.output, None, loss_param)
    model_recurrent.add_loss(loss)

    model_recurrent.compile(optimizer=optimizer)

    # model_recurrent.summary()

    early_stopping = EarlyStopping(monitor="loss",
                                   patience=10, min_delta=1e-4, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="loss",
                                  factor=0.5, patience=2, min_delta=1e-4, verbose=0)

    callbacks = [early_stopping, reduce_lr]

    # Fit the model.
    model_recurrent.fit(x=xtrain, batch_size=batch_size, epochs=epochs,
                        validation_data=[xtest], verbose=1, callbacks=callbacks)

    print("Finished running deep hedging algorithm!")

    call = EuropeanCall()

    price_BS = call.get_BS_price(S=S_test[0], sigma=sigma,
                                 risk_free=risk_free, dividend=dividend, K=strike,
                                 exercise_date=maturity_date,
                                 calculation_date=calculation_date,
                                 day_count=day_count, dt=dt)
    delta_BS = call.get_BS_delta(S=S_test[0], sigma=sigma,
                                 risk_free=risk_free, dividend=dividend, K=strike,
                                 exercise_date=maturity_date,
                                 calculation_date=calculation_date,
                                 day_count=day_count, dt=dt)

    PnL_BS = call.get_BS_PnL(S=S_test[0],
                             payoff=payoff_func(S_test[0][:, -1]), delta=delta_BS,
                             dt=dt, risk_free=risk_free,
                             final_period_cost=final_period_cost, epsilon=epsilon,
                             cost_structure=cost_structure)

    risk_neutral_price = \
        -option_payoff_test[0].mean() * np.exp(-risk_free * (N * dt))
    nn_simple_price = model_recurrent.evaluate(xtest, batch_size=test_size, verbose=0)

    print("The Black-Scholes model price is %2.3f." % price_BS[0][0])
    print("The Risk Neutral price is %2.3f." % risk_neutral_price)
    print("The Deep Hedging (with simple network) price is %2.3f." % nn_simple_price)

    try:
        nn_recurrent_price = model_recurrent.evaluate(xtest, batch_size=test_size, verbose=0)
        print("The Deep Hedging (with recurrent network) price is %2.3f." % nn_recurrent_price)
    except:
        print("No Recurrent model.")

    print("Plotting PnL")
    bar1 = PnL_BS + price_BS[0][0]
    bar2 = model_recurrent(xtest).numpy().squeeze() + price_BS[0][0]

    # Plot Black-Scholes PnL and Deep Hedging PnL (with BS_price charged on both).
    fig_PnL = plt.figure(dpi=125, facecolor='w')
    fig_PnL.suptitle("Black-Scholes PnL vs Deep Hedging PnL \n",
                     fontweight="bold")
    ax = fig_PnL.add_subplot()
    ax.set_title("Simple Network Structure with epsilon = " + str(epsilon),
                 fontsize=8)
    ax.set_xlabel("PnL")
    ax.set_ylabel("Frequency")
    ax.hist((bar1, bar2), bins=30,
            label=["Black-Scholes PnL", "Deep Hedging PnL"])
    ax.legend()
    # plt.show()

    if args.figname == "Default":
        figname = "%i_%i_%i_%s_%s" % (m, d, maxT, str(epsilon), args.input_model)
    else:
        figname = args.figname

    if args.figname == "Default":
        outdir = os.getcwd() + "/"
    else:
        outdir = args.outdir

    plt.savefig(outdir + figname + "_PnL.png")

    output = pd.Series()


    def cvar(wealth, w, alpha):
        return np.mean(w + (np.maximum(-wealth - w, 0) / (1 - alpha)))


    Var = model_recurrent(xtest).numpy().squeeze()
    output['d'] = d
    output['m'] = m
    output['maxT'] = maxT
    output['model'] = args.input_model
    output['epsilon'] = epsilon
    output['CVar99'] = cvar(Var, risk_neutral_price, 0.99)
    output['CVar95'] = cvar(Var, risk_neutral_price, 0.95)
    output['CVar90'] = cvar(Var, risk_neutral_price, 0.90)
    output['CVar80'] = cvar(Var, risk_neutral_price, 0.80)
    output['CVar50'] = cvar(Var, risk_neutral_price, 0.50)

    output['Var99'] = np.quantile(Var, 0.99)
    output['Var95'] = np.quantile(Var, 0.95)
    output['Var90'] = np.quantile(Var, 0.90)
    output['Var80'] = np.quantile(Var, 0.80)
    output['Var50'] = np.quantile(Var, 0.50)

    output['Mean_PnL'] = np.mean(Var)
    output['Std_PnL'] = np.std(Var)

    output['price'] = nn_simple_price
    output['price_BS'] = price_BS[0][0]
    output['price_free'] = risk_neutral_price

    output.to_csv(outdir + figname + "_pd.csv")

    pd.DataFrame(Var).to_csv(outdir + figname + "_Var.csv")

    ii = False
    if ii:

        print("Plotting Wealth")

        model = model_recurrent
        change_wealth = list()
        for w in tqdm(range(N + 2)):
            # print("looking for wealth d=" + w, end='\r')

            intermediate_layer_model = Model(inputs=model.input,
                                             outputs=model.get_layer("wealth_%i" % w).output)
            change_wealth.append(intermediate_layer_model.predict(xtest))

        wealths = pd.DataFrame(np.array(change_wealth)[:, :, 0])
        fig, ax = plt.subplots()

        # ax.plot(options[0],label='options')
        ax.plot(wealths.iloc[:, :], label='wealth')

        ax.set(ylabel='Wealth', xlabel='Days', title='Wealth movement')
        plt.savefig(outdir + figname + "_Wealth.png")

    print("Plotting Deltas")

    days_from_today = 15
    tau = (N - days_from_today) * dt

    min_S = S_test[0][:, days_from_today].min()
    max_S = S_test[0][:, days_from_today].max()

    S_range = np.linspace(min_S * 0.6, max_S * 1.4, 101)

    in_sample_range = S_range[np.any([S_range >= min_S, S_range <= max_S], axis=0)]
    out_sample_range_low = S_range[S_range < min_S]
    out_sample_range_high = S_range[S_range > max_S]

    # Attention: Need to transform it to be consistent with the information set.
    if information_set == "S":
        I_range = S_range  # Information set
    elif information_set == "log_S":
        I_range = np.log(S_range)
    elif information_set == "normalized_log_S":
        I_range = np.log(S_range / S0)
    else:
        raise Exception("There is a bug in my code, invalid information_set, yet it should have been taken care of by "
                        "the parser")
    # Compute Black-Scholes delta for S_range.
    # Reference: https://en.wikipedia.org/wiki/Greeks_(finance)
    d1 = (np.log(S_range) - np.log(strike) + \
          (risk_free - dividend + (sigma ** 2) / 2) * tau) \
         / (sigma * np.sqrt(tau))

    model_delta = norm.cdf(d1) * np.exp(-dividend * tau)

    model = model_recurrent
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer("delta_%i" % days_from_today).output)

    if "CLAMP" in args.input_model:
        inputs = [Input(1, ), Input(1, )]
        intermediate_inputs = Concatenate()(inputs)

        outputs = model.get_layer("delta_" + str(days_from_today))(intermediate_inputs
                                                                   , model_delta.astype(np.float32))
        MODEL = Model(inputs, outputs)

        nn_delta = MODEL([I_range, I_range])

    elif ("TCN" in args.input_model) | ("LSTM" in args.input_model):
        # inputs = list()
        # inW = list()
        # for m in range(maxT):
        #     inputs.append(Input(1,))
        #     inW.append(I_range)
        intermediate_inputs = Input(101,2,maxT)

        outputs = model.get_layer("delta_" + str(days_from_today))(intermediate_inputs)

        MODEL = Model(inputs, outputs)

        nn_delta = MODEL([[I_range, I_range], [I_range, I_range]])

    else:
        inputs = [Input(1, ), Input(1, )]
        intermediate_inputs = Concatenate()(inputs)

        outputs = model.get_layer("delta_" + str(days_from_today))(intermediate_inputs)
        MODEL = Model(inputs, outputs)

        nn_delta = MODEL([I_range, I_range])

    pd.DataFrame(nn_delta).to_csv(outdir + figname + "_delta.csv")
    # Create a plot of Black-Scholes delta against deep hedging delta.
    fig_delta = plt.figure(dpi=125, facecolor='w')
    fig_delta.suptitle("Black-Scholes Delta vs Deep Hedging Delta \n", \
                       fontweight="bold")
    ax_delta = fig_delta.add_subplot()
    ax_delta.set_title("Simple Network Structure with " +
                       "t=" + str(days_from_today) + ", " +
                       "epsilon=" + str(epsilon),
                       fontsize=8)
    ax_delta.set_xlabel("Price of the Underlying Asset")
    ax_delta.set_ylabel("Delta")
    ax_delta.plot(S_range, model_delta, label="Black-Scholes Delta")
    ax_delta.scatter(in_sample_range, nn_delta[np.any([S_range >= min_S, S_range <= max_S], axis=0)], c="red", s=2,
                     label="In-Range DH Delta")
    ax_delta.scatter(out_sample_range_low, nn_delta[S_range < min_S], c="green", s=2, label="Out-of-Range DH Delta")
    ax_delta.scatter(out_sample_range_high, nn_delta[S_range > max_S], c="green", s=2)

    ax_delta.legend()

    plt.savefig(outdir + figname + "_delta_15.png")
