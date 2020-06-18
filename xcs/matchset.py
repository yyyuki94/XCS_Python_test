import sys, random, itertools, copy
import numpy as np

from abc import ABCMeta, abstractmethod

from xcs.population import Population
from xcs.classifier import Classifier


class MatchSet:
    def __init__(self, population: Population, sigma: np.ndarray, theta_mna: int, P_s: float, time):
        self.M = []
        self.act_min = population.act_min
        self.act_max = population.act_max
        while len(self.M) == 0:
            for cl in population:
                if self.__does_match(cl, sigma):
                    self.M.append(cl)
            if self.__unique_act().shape[0] < theta_mna:
                cl_c = self.__gen_covering_clf(sigma, P_s, time)
                population.append(cl_c)
                population.delete_from_population()
                self.M = []

    def __iter__(self):
        self.__idx_current = 0
        return self

    def __next__(self):
        if self.__idx_current == len(self):
            raise StopIteration()

        idx = self.__idx_current
        self.__idx_current += 1

        return self.M[idx]

    def __getitem__(self, idx):
        return self.M[idx]

    def __len__(self):
        return len(self.M)

    def __does_match(self, cl, sigma):
        for x_cl, x_s in zip(cl["condition"], sigma):
            if (x_cl != 2) and (x_cl != x_s):
                return False
        return True

    def __gen_covering_clf(self, sigma, P_s, time):
        acts = self.__unique_act().flatten()
        cl = Classifier(len(sigma), time)

        for i in range(len(cl["condition"])):
            if np.random.rand() < P_s:
                cl["condition"][i] = 2
            else:
                cl["condition"][i] = sigma[i]

        act_all = set(np.arange(self.act_min, self.act_max+1))
        act_tmp = list(act_all - set(acts))
        act_tmp = np.random.choice(act_tmp, 1)

        cl["action"] = act_tmp[0]

        return cl

    def __unique_act(self):
        acts = []
        for cl in self.M:
            acts.append(cl["action"])

        if len(acts) != 0:
            acts = np.unique(acts)

        return np.array(acts).flatten()

    def get_list_of_clfattr(self, key):
        tmp = []
        for i in range(len(self)):
            tmp.append(self[i][key])
        return np.array(tmp)

