import copy
import numpy as np

from abc import ABCMeta, abstractmethod

from xcs.actionset import ActionSet
from xcs.population import Population


class GAComponent(metaclass=ABCMeta):
    @abstractmethod
    def run_evolve(self, A: ActionSet, sigma, P: Population):
        pass

    @abstractmethod
    def add_time(self):
        self.t += 1


class SimpleGAComponent:
    def __init__(self, eps_0, theta_ga, theta_sub, chi, mu, do_ga_subsumption):
        self.eps_0 = eps_0
        self.theta_ga = theta_ga
        self.theta_sub = theta_sub
        self.chi = chi
        self.mu = mu
        self.do_ga_subsumption = do_ga_subsumption
        self.t = 0
        self.possible_act = None

    def __offspring(self, A: ActionSet):
        fitness_sum = 0
        for cl in A:
            fitness_sum += cl["fitness"]
        choice_point = np.random.rand() * fitness_sum
        fitness_sum = 0
        for cl in A:
            fitness_sum += cl["fitness"]
            if fitness_sum > choice_point:
                return cl

    def __apply_crossover(self, cl1, cl2):
        x = np.random.rand() * (len(cl1["condition"]))
        y = np.random.rand() * (len(cl1["condition"]))

        if x > y:
            x, y = y, x
        i = 0
        while True:
            if x <= i < y:
                cl1["condition"][i], cl2["condition"][i] = cl2["condition"][i], cl1["condition"][i]
            i += 1
            if i >= y:
                break

    def __apply_mutation(self, cl, sigma):
        i = 0
        while True:
            if np.random.rand() < self.mu:
                if cl["condition"][i] == 2:
                    cl["condition"][i] = sigma[i]
                else:
                    cl["condition"][i] = 2
                i += 1
            if i >= len(cl["condition"]):
                break
        if np.random.rand() < self.mu:
            select_idx = np.random.randint(0, len(self.possible_act))
            cl["action"] = self.possible_act[select_idx]

    def __could_subsume(self, cl):
        if cl["experience"] > self.theta_sub:
            if cl["error"] < self.eps_0:
                return True
        return False

    def __is_more_general(self, cl_gen, cl_spec):
        if np.sum(cl_gen["condition"] == 2) <= np.sum(cl_spec["condition"]):
            return False
        i = 0
        while True:
            if cl_gen["condition"][i] != 2 and cl_gen["condition"][i] != cl_spec["condition"][i]:
                return False
            i += 1

            if i < len(cl_gen["condition"]):
                break
        return True

    def __does_subsume(self, cl_sub, cl_tos):
        if cl_sub["action"] == cl_tos["action"]:
            if self.__could_subsume(cl_sub) and self.__is_more_general(cl_sub, cl_tos):
                return True

        return False

    def __insert_in_population(self, cl, P: Population):
        for c in P:
            if (c["condition"] == cl["condition"]).all() and c["action"] == cl["action"]:
                c["numerosity"] += 1
                return
        P.append(cl)

    def run_evolve(self, A: ActionSet, sigma, P: Population):
        sum_t = np.sum([cl["time_stamp"] * cl["numerosity"] for cl in A]) / np.sum([cl["numerosity"] for cl in A])
        self.possible_act = np.arange(P.act_min, P.act_max+1)

        if self.t - sum_t > self.theta_ga:
            for cl in A:
                cl["time_stamp"] = self.t
            parent_1 = self.__offspring(A)
            parent_2 = self.__offspring(A)
            child_1 = copy.deepcopy(parent_1)
            child_2 = copy.deepcopy(parent_2)
            child_1["numerosity"], child_2["numerosity"] = 1, 1
            child_1["experience"], child_2["experience"] = 0, 0
            if np.random.rand() < self.chi:
                self.__apply_crossover(child_1, child_2)
                child_1["prediction"] = (parent_1["prediction"] + parent_2["prediction"]) / 2
                child_1["error"] = (parent_1["error"] + parent_2["error"]) / 2
                child_1["fitness"] = (parent_1["fitness"] + parent_2["fitness"]) / 2
                child_2["prediction"] = child_1["prediction"]
                child_2["error"] = child_1["error"]
                child_2["fitness"] = child_2["fitness"]
            child_1["fitness"] = child_1["fitness"] * 0.1
            child_2["fitness"] = child_2["fitness"] * 0.1
            for child in (child_1, child_2):
                self.__apply_mutation(child, sigma)
                if self.do_ga_subsumption:
                    if self.__does_subsume(parent_1, child):
                        parent_1["numerosity"] += 1
                    elif self.__does_subsume(parent_2, child):
                        parent_2["numerosity"] += 1
                    else:
                        self.__insert_in_population(child, P)
                else:
                    self.__insert_in_population(child, P)
                P.delete_from_population()
