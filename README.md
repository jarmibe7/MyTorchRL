# Deep RL from Scratch
### ME 469: HW2
### Author: Jared Berry
### Due: 11/10/2025

Hello!

To test the neural network, run the following command from directory HW2/:

```

python test_nn.py

```

This will test the custom neural network architecture on a simple classification problem. The data will be visualized after running this command, and closing the plot will run the classification. The accuracy will print to the terminal. To change the
dataset, switch the CSV file being used in test_nn.py. Valid datasets are in test_data/. Performance here will vary depending on the network architecture specified in test_nn.py.

To run RL training, run the following command from directory HW2/:

```

python run.py

```

Training can be done with tabular vanilla Q-Learning or with a deep formulation of the Advantage Actor-Critic (A2C) algorithm. This can
be selected by setting the variable ```use_deep``` on line 40 in run.py.

#### Code Structure

- data/: Contains data from ds1.

- figures/: Contains figures demonstrating training progress.

- gym/: Contains files for gymnasium-like environment.

- metrics/: Contains JSON files with metric data for Vanilla QL.

- my_torch/: Contains files for neural net architecture.

- test_data/: Contains test classification datasets, provided by COMP_SCI 349: Machine Learning.

- a2c.py: File containing deep A2C algorithm for RL.

- q_learning.py: File containing Vanilla Q-Learning for RL.

- run.py: Contains main function for running RL training.

- test_nn.py: Script for testing custom neural net functionality.

- utils.py: Various utility functions.