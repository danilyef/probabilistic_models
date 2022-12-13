import numpy as np


def sample_categorical(dist: np.ndarray) -> np.int64:
    """
    Draws a single sample from a categorical distribution.
    :param dist: NumPy array listing the probability of each outcome.
    :returns: Index of the sampled element.
    """

    assert type(dist) == np.ndarray
    assert dist.ndim == 1
    assert np.isclose(dist.sum(), 1)

    return np.random.choice(len(dist), p=dist)


def kld(p, q):
    """
    Computes the Kullback-Leibler divergence between p and q.
    :param p: true distribution
    :param q: estimated distribution
    :return: Kullback-Leibler Divergence between p and q
    """
    return (p * np.log(p / (q + 1e-10))).sum()  # add a small constant for numeric stability


def approx_error(bayes_net, approx_function, exact, query_variable, evidence, sample_counts, n_runs=100, **kwargs):
    """
    Computes the approximation error for a given approximation method.
    """
    mean_errors = []
    
    for num_samples in sample_counts:
        estimates = np.array([approx_function(bayes_net, query_variable, evidence, num_samples, **kwargs) for i in range(n_runs)])
        mean_errors.append(kld(exact, estimates) / len(estimates))
    return mean_errors