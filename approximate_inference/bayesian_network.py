from typing import List, Optional, Callable, Tuple, Iterator

import numpy as np


class Variable:

    def __init__(self, pdt: np.ndarray, idx_mapping: List[int]) -> None:
        """
        Creates a Variable object which is used to build Bayesian Networks.

        :param pdt: expanded and sorted (conditional) probability distribution table.
        :param idx_mapping: mapping of dimension to variable index.
         idx_mapping[0] == variable.id, idx_mapping[1:] == parents

        :returns: A Variable object.
        """
        # basic info
        assert len(idx_mapping) >= 1, 'Variable must have an id!'
        if idx_mapping[0] in idx_mapping[1:]:
            raise UserWarning(f'It makes no sense to condition on self e.g. P(A | A)! ID: {idx_mapping[0]}')
        self.id = idx_mapping[0]
        self.parents = set(idx_mapping[1:])
        self.children = set()
        self.pdt = pdt
        self.num_nodes = len(pdt.shape)
        self.num_values = pdt.shape[idx_mapping[0]]

        # resampling distribution, parents
        self.resampling_pdt = None
        self.resampling_parents = None

    def __call__(self, sample: np.ndarray, resampling: bool=False) -> np.ndarray:
        """
        Returns the probability distribution over the variable, given its parents or given its Markov blanket.

        :param sample: A NumPy array holding the values of the parent variables sorted by variable id.
                       Values of non-parent variables will be ignored.
        :param resampling: If False, P(X|pa(X)) will be returned. Otherwise P(X|mb(X)).
        :returns: A NumPy array representing the probability distribution over the variable,
                  given its parents or given its markov blanket.
        """

        assert len(sample) == self.num_nodes, f'Size of sample must be equal to number of variables in the Network. ' \
                                              f'Given: {len(sample)}, Expected: {self.num_nodes}'

        if resampling:
            assert self.resampling_parents is not None, 'Resampling distribution not computed!'
            parents = self.resampling_parents
            pdt = self.resampling_pdt
        else:
            parents = self.parents
            pdt = self.pdt

        index = ()
        for i in range(self.num_nodes):
            if i == self.id:
                index = index + (slice(None),)
            elif i in parents:
                index = index + (sample[i],)
            else:
                index = index + (0,)
        return pdt[index]


class BayesNet:

    def __init__(self, *pdt_ids_tuples: Tuple[np.ndarray, List[int]],
                 resampling_distribution: Optional[
                     Callable[[Variable, 'BayesNet'],
                              Tuple[np.ndarray, List[int]]]] = None) -> None:
        """
        Creates a BayesNet object.

        :param pdt_ids_tuples: Arbitrarily many tuples in format (np.ndarray, [id1, id2, ...]).
            Each tuple defines one variable of the Bayesian Network. The numpy array stacks
            the Probability Distribution Tables (PDTs) of the variable conditioned on all value
            combinations of its parents. The integer list denotes the variable's id followed by
            its parent variable ids (if any), matching the order of dimensions in the PDTs.
            Each variable id is the index of the column in the data the variable corresponds to.
        :param resampling_distribution: Callable computing the resampling distribution given
            a variable and a BayesNet (Only needed in PS 3, Assignment 'Gibbs Sampling', and
            is described there thoroughly, completely ignore it otherwise).
        :return: The BayesNet object.
        """
        self.nodes = dict()
        self.pdts, self.indices = zip(*pdt_ids_tuples)
        num_nodes = len(self.pdts)

        for pdt, structure in zip(self.pdts, self.indices):
            assert type(pdt) == np.ndarray, f'Probability Density Table has to be a NumPy ndarray' \
                                            f' but was of type {type(pdt)}!'
            assert np.all(np.isclose(pdt.sum(axis=0), 1)), f'Probabilities on axis 0 have to sum to 1!'
            assert pdt.ndim == len(
                structure), f'Number of table dimensions has to match ' \
                            f'the number of Variable indices (1 (self) + n_parents)!' \
                            f'N-Dimensions: {pdt.ndim} != Len(Idcs): {len(structure)}!'
            # Order PDT dimensions by variable id
            pdt = pdt.transpose(np.argsort(structure))
            # Add singleton dimensions for all other variables
            to_expand = tuple(set(range(num_nodes)).difference(set(structure)))
            pdt = np.expand_dims(pdt, axis=to_expand)
            # Create Variable object
            self.nodes[structure[0]] = Variable(pdt, structure)

        # Set children of nodes
        for node in self.nodes.values():
            for parent_id in node.parents:
                self.nodes[parent_id].children.add(node.id)

        # Compute resampling distributions
        if resampling_distribution is not None:
            for node in self.nodes.values():
                node.resampling_pdt, node.resampling_parents = resampling_distribution(node, self)

    def __len__(self) -> int:
        """
        Retrieves the number of Variables in the network.

        :return: Variable count as integer
        """
        return len(self.nodes)

    def __getitem__(self, id: int) -> Variable:
        """
        Retrieves a Variable based on its id.

        :param id: Id of the Variable.
        :return: A BayesNet Variable Object with the corresponding id.
        :raises KeyError: if id is not found.
        """
        return self.nodes[id]

    def __iter__(self) -> Iterator[Variable]:
        """
        Iterates over all variables in the network in topological order.

        :yields: Variable after Variable according to the Network's topology.
        """

        def __topological_sort__(v: int):
            visited[v] = True
            for i in self.nodes[v].children:
                if not visited[i]:
                    __topological_sort__(i)
            stack.append(v)

        visited = [False] * len(self)
        stack = []
        for i in range(len(self)):
            if not visited[i]:
                __topological_sort__(i)

        for i in stack[::-1]:
            yield self[i]
