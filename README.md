# Deep Hedging 
## Pricing Options (Derivatives) using Deep Learning models

The Framework is composed of a main, which either loads data (real datas), or generate datas (according to the Brownian motion). The input datas are stock prices, and an information set (which can be any information a human trader might use).
For a European option, the set of relevant information at time t are the log-moneyness, the time to maturity, and the volatility.
The model, defined in the deep_hedging_models files, takes the inputs, and simulate a N days trading days. Each day, the Agent infer a strategy and takes an action (see ![Model Architecture](https://gitlab.ethz.ch/dbolli/deep-hedging/-/blob/1410baa2d43cedafc0a38c8a54c01351e2f30afb/Deep-learning_strategy.jpg)).\\


To run a model, you can simply run the main_Deep_Hedging.py, and specify the following arguments:
![main arguments](https://gitlab.ethz.ch/dbolli/deep-hedging/-/blob/master/main_args.jpg)
Other options such as the volatility of the generated datas, or the strike price can be set at the begining of the main.\\

![](deep-hedging/-/blob/master/main_args.jpg)

