#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 2021-12-01
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

from .minimization import MinimizationAlgorithm
from .test_functions import Function

import numpy as np


COMMA_SELECTION = 'comma-selection'
PLUS_SELECTION = 'plus-selection'

STRATEGIES = [COMMA_SELECTION, PLUS_SELECTION]


class EvolutionalStrategy(MinimizationAlgorithm):
    def __init__(self,
                 target: Function,
                 mu: int,  # number of parents
                 rho: int,  # number of parents involved in the procreation of an offspring
                 lmb: int,  # number of offspring
                 strategy: str,  # comma-selection, plus-selection
                 mutation_std: float = 0.1,
                 ):
        assert 1 < mu <= lmb
        assert 1 <= rho <= mu
        assert strategy in STRATEGIES
        super(EvolutionalStrategy, self).__init__(target)

        self.mu = mu
        self.rho = rho
        self.lmb = lmb
        self.strategy = strategy
        self.mutation_std = mutation_std

        self.population = np.random.uniform(
            low=self.target.bounds[:, 0],
            high=self.target.bounds[:, 1],
            size=(self.lmb if strategy == COMMA_SELECTION else self.lmb + self.mu, 2),
        )

    def fit(self, num_iterations: int = 1000, verbose=False):
        metrics = []
        units = []
        for i in range(num_iterations):
            curr_metrics, unit = self._iteration_step(verbose)
            metrics.append(curr_metrics)
            units.append(unit)
        return np.vstack(metrics), np.vstack(units)

    def _iteration_step(self, verbose):
        # calculating loss function
        fitness = self.target(*self.population.T)
        # selecting parents
        if verbose:
            print(f'[*] Selecting parents.')
        parents_idx = np.argsort(fitness)[:self.mu]
        parents = self.population[parents_idx]

        if verbose:
            print(f'[*] Got parents: {parents}.')
            print(f'[*] Generating offsprings.')
        offsprings = self._generate_offsprings(parents)

        if verbose:
            print(f'[*] Got offsprings: {offsprings}')
            print(f'[*] Applying mutation.')
        mutated = self._mutation(offsprings)

        if self.strategy == PLUS_SELECTION:  # plus-selection
            self.population = np.vstack([mutated, parents], )
        elif self.strategy == COMMA_SELECTION:  # comma-selection
            self.population = mutated
        else:
            raise ValueError(f'Strategy should be one of {STRATEGIES}.')

        return np.array([fitness.mean(), fitness[parents_idx[0]]]), parents[0]

    def _generate_offsprings(self, parents: np.ndarray) -> np.ndarray:
        offsprings = np.empty((self.lmb, 2))
        for i in range(self.lmb):
            curr_parents_idx = np.random.choice(self.mu, self.rho)
            curr_parents = parents[curr_parents_idx]
            offsprings[i] = curr_parents.mean(axis=0)
        return offsprings

    def _mutation(self, offsprings: np.ndarray) -> np.ndarray:
        for i in range(offsprings.shape[0]):
            offsprings[i] += np.random.normal(loc=0, scale=self.mutation_std, size=offsprings.shape[1])
        return offsprings
