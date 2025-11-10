# Robot Navigation
### ME 469: HW2
### Author: Jared Berry
### Due: 11/10/2025

Hello!

To run for submission A, run the following command from directory HW2/:

```

python test_nn.py

```

This will test the custom neural network architecture on a simple classification problem. The data will be visualized after running this command, and closing the plot will run the classification. The accuracy will print to the terminal. To change the
dataset, switch the CSV file being used in test_nn.py. Valid datasets are in test_data/. Performance here will vary depending on the network architecture specified in test_nn.py.

To run a test of the custom Gymnasium environment, run the following command from directory HW2/:

```

python test_gym.py

```

To run RL training, run the following command from directory HW2/:

```

python run.py

```

Training is unlikely to converge, but will run.

#### Code Structure

- gym/: Contains files for gymnasium-like environment

- my_torch/: Contains files for neural net architecture

- data/: Contains data from ds1.

- test_data/: Contains test classification datasets, provided by COMP_SCI 349: Machine Learning.

- a2c.py: Main file containing A2C algorithm for RL.

- run.py: Contains main function for running RL training.

- test_gym.py: Script for testing custom Gymnasium functionality.

- test_nn.py: Script for testing custom neural net functionality.

- utils.py: Various utility functions.