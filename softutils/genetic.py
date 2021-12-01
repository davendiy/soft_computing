#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 2021-11-30
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

import numpy as np
import bitstring
from itertools import combinations

# from tqdm.notebook import trange

from .test_functions import Function
from .minimization import MinimizationAlgorithm

PANMIXION    = 'panmixion'
SELECTION    = 'selection'
INBREEDING   = 'inbreeding'
OUTBREEDING  = 'outbreeding'
PROPORTIONAL = 'proportional'

PARENTS_CHOICE = [
    PANMIXION, SELECTION,
    INBREEDING, OUTBREEDING,
    PROPORTIONAL,
]

ELITE     = 'elite'
EXTRUSION = 'extrusion'

SURVIVING_TYPE = [
    ELITE, EXTRUSION
]


class GeneticAlgorithm(MinimizationAlgorithm):

    def __init__(self, target: Function,
                 population_size: int, parents_amount: int,
                 include_parents=True,
                 choice_type=PANMIXION, surviving_type=ELITE,
                 eps=10e-4, mutation_prob=0.5):

        self.pop_size = population_size
        self.parents_amount = parents_amount
        self.choice_type = choice_type
        self.surviving_type = surviving_type
        self.eps = eps
        self.include_parents = include_parents
        self.mutation_prob = mutation_prob
        super(GeneticAlgorithm, self).__init__(target)

        partitions = []
        digits = []
        for (x_left, x_right) in self.target.bounds:
            partition_amount = (x_right - x_left) / eps
            partitions.append(partition_amount)
            digits.append(int(np.log2(partition_amount)) + 2)

        self.partitions = np.array(partitions)
        self.digits = digits

        self.population = np.random.randint(
            low=0,
            high=np.max(self.partitions),
            size=(self.pop_size, self.target.dim),
        )

    def _el2point(self, element):
        element = element % self.partitions
        element = self.target.bounds[:, 0] + self.eps * element
        return element

    def _calculate_loss(self, population):
        values = []
        for el in population:
            values.append(self._el2point(el))
        values = np.array(values)
        return self.target(*values.T)

    def fit(self, num_iterations, verbose=False) -> (np.ndarray, np.ndarray):
        metrics = []
        min_trace = []
        for i in range(num_iterations):
            cur_metrics, best_unit = self._iteration_step(verbose=verbose)
            metrics.append(cur_metrics)
            min_trace.append(best_unit)
        return np.vstack(metrics), np.vstack(min_trace)

    def _iteration_step(self, verbose):
        if verbose:
            print(f'[*] Choosing parents using {self.choice_type} method.')
        parents = self._parents_choice()
        if verbose:
            print(f"[*] Parents: {parents}")
            print(f'[*] Applying crossover.')
        offsprings = self._crossover(parents)
        if verbose:
            print(f'[*] Offsprings: {offsprings}')
            print(f'[*] Applying mutation.')
        offsprings = self._mutation(offsprings)
        if verbose:
            print(f'[*] Offsprings: {offsprings}')
            print(f'[*] Selecting best childrens using {self.surviving_type} method.')
        self.population = self._get_survivors(offsprings)
        average_loss = self._calculate_loss(self.population).mean()
        best_unit = self._el2point(self.population[0])
        best_loss = self.target(*best_unit)
        if verbose:
            print(f'[*] Next population: {self.population}')
            print(f'[*] Best loss: {best_loss}')
            print(f'[*] Average loss: {average_loss}')
            print(f'[*] Best unit: {best_unit}')
        return np.array([best_loss, average_loss]), best_unit

    def _parents_choice(self):
        if self.choice_type == PANMIXION:

            return self.population[np.random.choice(range(self.pop_size), size=self.parents_amount)]
        elif self.choice_type == SELECTION:
            loss = self._calculate_loss(self.population)
            return self.population[loss < loss.mean()][:self.parents_amount]
        elif self.choice_type == INBREEDING:
            first = self.population[np.random.choice(range(self.pop_size))]
            diffs = np.abs(first - self.population)
            indx = np.argsort(np.sum(diffs, axis=1))[::-1] + 1
            probs = indx / indx.sum()
            res_indx = np.random.choice(range(self.pop_size), p=probs, size=self.parents_amount)
            return self.population[res_indx]
        elif self.choice_type == OUTBREEDING:
            first = self.population[np.random.choice(range(self.pop_size))]
            diffs = np.abs(first - self.population)
            indx = np.argsort(np.sum(diffs, axis=1)) + 1
            probs = indx / indx.sum()
            res_indx = np.random.choice(range(self.pop_size), p=probs, size=self.parents_amount)
            return self.population[res_indx]
        elif self.choice_type == PROPORTIONAL:
            probs = self._calculate_loss(self.population)
            probs -= probs.min()
            probs /= probs.sum()
            res_indx = np.random.choice(range(self.pop_size), p=probs, size=self.parents_amount)
            return self.population[res_indx]
        else:
            raise ValueError(f"Unknown choice type. Should be one of {PARENTS_CHOICE}")

    def _crossover(self, parents: np.ndarray) -> np.ndarray:
        offsprings = []
        for par1, par2 in combinations(parents, 2):
            son1 = []
            son2 = []
            for i, (x1, x2) in enumerate(zip(par1, par2)):
                x1 = bitstring.BitArray(int=int(x1 % self.partitions[i]),
                                            length=int(self.digits[i]))
                x2 = bitstring.BitArray(int=int(x2 % self.partitions[i]),
                                        length=int(self.digits[i]))

                k = np.random.randint(0, self.digits[i]-1)
                son1.append((x1[:k] + x2[k:]).int % self.partitions[i])
                son2.append((x2[:k] + x1[k:]).int % self.partitions[i])
            offsprings.append(np.array(son1))
            offsprings.append(np.array(son2))
        if self.include_parents:
            offsprings += list(parents)
        return np.array(offsprings)

    def _mutation(self, offsprings: np.ndarray) -> np.ndarray:
        mutants = []
        for el in offsprings:
            el = el.copy()
            if np.random.uniform() > self.mutation_prob:
                for i in range(len(el)):
                    x = bitstring.BitArray(int=int(el[i] % self.partitions[i]), length=int(self.digits[i]))
                    k = np.random.randint(0, self.digits[i] - 1)
                    x[k] = 1 - x[k]
                    el[i] = x.int % self.partitions[i]
            mutants.append(el)
        return np.array(list(offsprings) + mutants)

    def _get_survivors(self, offsprings):
        if self.surviving_type == ELITE:
            loss = self._calculate_loss(offsprings)
            indx = np.argsort(loss)[:self.pop_size]
            return offsprings[indx]
        else:
            raise NotImplementedError()
