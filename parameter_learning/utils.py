import numpy as np
from bayesian_network import BayesNet, Variable

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

def sample_forward(bayes_net: BayesNet, sample_size: int) -> np.ndarray:
    '''
    Samples from the full joint distribution.
    :param bayes_net: A Bayesian network of type BayesNet.
    :param sample_size: The number of samples to draw from the Bayesian network.
    :returns: A NumPy array of type np.int64 with shape (sample_size, len(bayes_net))
              containing samples from the Bayesian network
    '''
    
    # array holding the samples
    samples = np.empty((sample_size, len(bayes_net)), np.int64)
    
    # do a forward pass for each sample
    for i in range(sample_size):
        # iterate over variables in topological order
        for variable in bayes_net:
            # get the distribution of variable given value of parents
            distribution = variable(samples[i])
            # store value in sample
            samples[i, variable.id] = sample_categorical(distribution)
    
    return samples


def sample_lw(bayes_net: BayesNet, sample_size: int, evidence: dict={}):
    '''
    Samples from the mutilated network.
    :param bayes_net: A Bayesian network of type BayesNet.
    :param sample_size: The number of samples to draw from the Bayesian network.
    :param evidence: A dictionary of evidence variables (keys, int) and their correponding values (values, int).
    :returns: A NumPy array of type np.int64 with shape (sample_size, len(bayes_net)),
              and a NumPy array of shape (sample_size,) with the corresponding weight for each sample.
    '''
    
    # arrays holding the samples and the weights
    samples = np.empty((sample_size, len(bayes_net)), np.int64)
    weights = np.ones(sample_size)
    
    ### BEGIN SOLUTION
    
    # set evidence
    for e in evidence:
        samples[:, e] = evidence[e]
        
    # do forward pass for each sample
    for i in range(sample_size):
        # iterate over variables in topological order
        for variable in bayes_net:
            # get the distribution of variable given value of parents
            distribution = variable(samples[i])
            if variable.id in evidence:
                # if variable in evidence, update the weight
                weights[i] *= distribution[evidence[variable.id]]
            else:
                # else, sample a value according to distribution
                samples[i, variable.id] = sample_categorical(distribution)
    ### END SOLUTION
    
    return samples, weights

def get_default_bayes_net(resampling_distribution=None):
    _A_, _B_, _C_, _D_, _E_ = 0, 1, 2, 3, 4

    A = np.array([0.2, 0.8])
    B_A = np.array([[0.9, 0.2], [0.1, 0.8]])
    C = np.array([0.9, 0.1])
    D_BC = np.array([[[0.1, 0.2], [0.99, 0.8]], [[0.9, 0.8], [0.01, 0.2]]])
    E_C = np.array([[0.7, 0.4], [0.3, 0.6]])

    return BayesNet(
        (A, [_A_]),
        (B_A, [_B_, _A_]),
        (C, [_C_]),
        (D_BC, [_D_, _B_, _C_]),
        (E_C, [_E_, _C_]),
        resampling_distribution=resampling_distribution
    )
