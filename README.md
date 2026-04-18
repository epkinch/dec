# Deep Embedded Clustering

This package implements the algorithm described in paper "Unsupervised Deep Embedding for Clustering Analysis". It depends on {}.

This implementation is intended for reproducing the results in the paper using PyTorch.

## Usage

To compare the hyperparamater vs accuracy of our DEC model and KMeans, Spectral Clustering, and DEC w/o backprop you first must compute the values for DEC and DEC w/o backprop. To compute accuracies for DEC, run dec.py with a command line argument of 1. This will take significant compute. The results are saved to accuracies.csv. To run within backprop, you must change a hardcoded 'backprop' variable to false on line 383.

To get the hyperparameter vs accuracy datapoints for MNIST and Spectral Clustering, run replication.py. All accuracies are saved in the same csv.

To graph your results, run graphy.py. You should get a figure 

### Docker


