from tensorflow.keras.layers import Input, Dense, Concatenate, Subtract, \
    Lambda, Add, Dot, BatchNormalization, Activation, LeakyReLU, LSTM, \
    LayerNormalization, Dropout, MultiHeadAttention, Conv1D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import he_normal, Zeros, he_uniform, TruncatedNormal
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

intitalizer_dict = {
    "he_normal": he_normal(),
    "zeros": Zeros(),
    "he_uniform": he_uniform(),
    "truncated_normal": TruncatedNormal()
}

bias_initializer = he_uniform()


class Strategy_Layer(tf.keras.layers.Layer):
    def __init__(self, d=None, m=None, use_batch_norm=None, \
                 kernel_initializer="he_uniform", \
                 activation_dense="relu", activation_output="linear",
                 delta_constraint=None, day=None):
        super().__init__(name="delta_" + str(day))
        self.d = d
        self.m = m
        self.use_batch_norm = use_batch_norm
        self.activation_dense = activation_dense
        self.activation_output = activation_output
        self.delta_constraint = delta_constraint
        self.kernel_initializer = kernel_initializer

        self.intermediate_dense = [None for _ in range(d)]
        self.intermediate_BN = [None for _ in range(d)]

        for i in range(d):
            self.intermediate_dense[i] = Dense(self.m,
                                               kernel_initializer=self.kernel_initializer,
                                               bias_initializer=bias_initializer,
                                               use_bias=(not self.use_batch_norm))
            if self.use_batch_norm:
                self.intermediate_BN[i] = BatchNormalization(momentum=0.99, trainable=True)

        self.output_dense = Dense(1,
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=bias_initializer,
                                  use_bias=True)

    def call(self, input):

        for i in range(self.d):
            if i == 0:
                output = self.intermediate_dense[i](input)
            else:
                output = self.intermediate_dense[i](output)

            if self.use_batch_norm:
                # Batch normalization.
                output = self.intermediate_BN[i](output, training=True)

            if self.activation_dense == "leaky_relu":
                output = LeakyReLU()(output)
            else:
                output = Activation(self.activation_dense)(output)

        output = self.output_dense(output)

        if self.activation_output == "leaky_relu":
            output = LeakyReLU()(output)
        elif self.activation_output == "sigmoid" or self.activation_output == "tanh" or \
                self.activation_output == "hard_sigmoid":
            # Enforcing hedge constraints
            if self.delta_constraint is not None:
                output = Activation(self.activation_output)(output)
                delta_min, delta_max = self.delta_constraint
                output = Lambda(lambda x: (delta_max - delta_min) * x + delta_min)(output)
            else:
                output = Activation(self.activation_output)(output)

        return output


def Deep_Hedging_Model_Transformer(N=None, d=None, m=None, \
                                   risk_free=None, dt=None, initial_wealth=0.0, epsilon=0.0, maxT=5, \
                                   final_period_cost=False, use_batch_norm=None, \
                                   kernel_initializer="he_uniform", \
                                   activation_dense="relu", activation_output="linear",
                                   delta_constraint=None, cost_structure="proportional", num_heads=256, dropout=0.2):
    # State variables.
    past_strategy = list()
    prc = Input(shape=(1,), name="prc_0")
    information_set = Input(shape=(1,), name="information_set_0")

    inputs = [prc, information_set]

    for j in range(N + 1):
        if j < N:
            # Define the inputs for the strategy layers.
            if j == 0:
                #   Strategy at t = -1 should be 0.
                strategy = Lambda(lambda x: x * 0.0)(prc)
            helper = Concatenate()([information_set, strategy])
            if j == 0:
                past_strategy.append(helper)
            past_strategy.append(helper)
            helper = past_strategy[-maxT:]
            helper1 = tf.stack(helper, axis=1)

            T = min(maxT, len(past_strategy))  # is set to maxT or less for the first time step

            for k in range(d):
                if k == d - 1:

                    x = LayerNormalization(epsilon=1e-6)(helper1)

                    x = MultiHeadAttention(
                        key_dim=T, num_heads=num_heads, dropout=dropout)(x, x)
                    x = Dropout(dropout)(x)
                    # Feed Forward Part
                    x = LayerNormalization(epsilon=1e-6)(x)
                    x = Conv1D(filters=T, kernel_size=1)(x)
                    x = Dropout(dropout)(x)
                    x = Conv1D(filters=helper1.shape[-1], kernel_size=1)(x)

            x = Reshape((x.shape[1] * x.shape[2],))(x)
            strategyhelper = Dense(1, kernel_initializer=kernel_initializer,
                                   bias_initializer=he_uniform(), activation=activation_output)(x)

            # delta_strategy = strategy_{t+1} - strategy_t
            if j == 0:
                delta_strategy = strategyhelper
            else:
                delta_strategy = Subtract(name="diff_strategy_" + str(j))([strategyhelper, strategy])

            if cost_structure == "proportional":
                # Proportional transaction cost
                absolutechanges = Lambda(lambda x: K.abs(x), name="absolutechanges_" + str(j))(delta_strategy)
                costs = Dot(axes=1)([absolutechanges, prc])
                costs = Lambda(lambda x: epsilon * x, name="cost_" + str(j))(costs)
            elif cost_structure == "constant":
                costs = Lambda(lambda x: epsilon + x * 0.0)(prc)

            if j == 0:
                wealth = Lambda(lambda x: initial_wealth - x, name="costDot_" + str(j))(costs)
            else:
                wealth = Subtract(name="costDot_" + str(j))([wealth, costs])

            # Wealth for the next period
            # w_{t+1} = w_t + (strategy_t-strategy_{t+1})*prc_t
            #         = w_t - delta_strategy*prc_t
            mult = Dot(axes=1)([delta_strategy, prc])
            wealth = Subtract(name="wealth_" + str(j))([wealth, mult])

            # Accumulate interest rate for next period.
            FV_factor = np.exp(risk_free * dt)
            wealth = Lambda(lambda x: x * FV_factor)(wealth)

            prc = Input(shape=(1,), name="prc_" + str(j + 1))
            information_set = Input(shape=(1,), name="information_set_" + str(j + 1))

            strategy = strategyhelper

            if j != N - 1:
                inputs += [prc, information_set]
            else:
                inputs += [prc]
        else:
            if final_period_cost:
                if cost_structure == "proportional":
                    # Proportional transaction cost
                    absolutechanges = Lambda(lambda x: K.abs(x), name="absolutechanges_" + str(j))(strategy)
                    costs = Dot(axes=1)([absolutechanges, prc])
                    costs = Lambda(lambda x: epsilon * x, name="cost_" + str(j))(costs)
                elif cost_structure == "constant":
                    costs = Lambda(lambda x: epsilon + x * 0.0)(prc)

                wealth = Subtract(name="costDot_" + str(j))([wealth, costs])
            # Wealth for the final period
            # -delta_strategy = strategy_t
            mult = Dot(axes=1)([strategy, prc])
            wealth = Add(name="wealth_" + str(j))([wealth, mult])

            # Add the terminal payoff of any derivatives.
            payoff = Input(shape=(1,), name="payoff")
            inputs += [payoff]

            wealth = Add(name="wealth_" + str(j + 1))([wealth, payoff])
    return Model(inputs=inputs, outputs=wealth)


def Delta_SubModel(model=None, days_from_today=None, share_stretegy_across_time=False, strategy_type="simple"):
    if strategy_type == "simple":
        inputs = model.get_layer("delta_" + str(days_from_today)).input
        intermediate_inputs = inputs
    elif strategy_type == "recurrent":
        inputs = [Input(1, ), Input(1, )]
        intermediate_inputs = Concatenate()(inputs)

    if not share_stretegy_across_time:
        outputs = model.get_layer("delta_" + str(days_from_today))(intermediate_inputs)
    else:
        outputs = model.get_layer("delta_0")(intermediate_inputs)

    return Model(inputs, outputs)
