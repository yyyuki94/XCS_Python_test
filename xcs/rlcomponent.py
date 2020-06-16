import sys, random, itertools, copy
import numpy as np

from abc import ABCMeta, abstractmethod

from xcs.actionset import ActionSet
from xcs.population import Population

# 強化学習部の抽象クラス
class RLComponent(metaclass=ABCMeta):    
    @abstractmethod
    def parameter_update(self, A: ActionSet, P: float, Pop: Population):
        pass

    
# Q学習ライクな強化学習部 
class QlearnLikeRLComponent(RLComponent):
    def __init__(self, theta_mna, P_s, p_explr, alpha,
                 beta, eps_0, nu, theta_sub, do_actionset_subsumption):
        self.theta_mna = theta_mna
        self.P_s = P_s
        self.p_explr = p_explr
        self.alpha = alpha
        self.beta = beta
        self.eps_0 = eps_0
        self.nu = nu
        self.theta_sub = theta_sub
        self.do_act_subsumption = do_actionset_subsumption
    
    def __update_fitness(self, A: ActionSet):
        sum_ac = 0
        acc_k = np.zeros(len(A))
        
        for i, cl in enumerate(A):
            if(cl["error"] < self.eps_0):
                acc_k[i] = 1
            else:
                acc_k[i] = self.alpha * (cl["error"] / self.eps_0) ** (- self.nu)
            sum_ac += acc_k[i] * cl["numeriosity"]
        for i, cl in enumerate(A):
            cl["fitness"] += self.beta * (acc_k[i] * cl["numeriosity"] / sum_ac - cl["fitness"])
    
    def __could_subsume(self, cl):
        if(cl["experience"] > self.theta_sub):
            if(cl["error"] < self.eps_0):
                return True
        return False
    
    def __is_more_general(self, cl_gen, cl_spec):
        if(np.sum(cl_gen["condition"] == 2) <= np.sum(cl_spec["condition"])):
            return False
        i = 0
        while True:
            if(cl_gen["condition"][i] != 2 and cl_gen["condition"][i] != cl_spec["condition"][i]):
                return False
            i += 1
            
            if(i < len(cl_gen["condition"])):
                break
        return True
    
    def __do_action_subsumption(self, A: ActionSet, P: Population):
        cl = None
        for c in A:
            if(self.__could_subsume(c)):
                if((cl is not None) or (np.sum(c["condition"]) == 2) > np.sum(c["condition"]) == 2) or \
                    ((np.sum(c["condition"]) == 2) == np.sum(c["condition"]) == 2) and (np.random.rand() < 0.5):
                    cl = c
        if(cl is not None):
            for c in A:
                if(self.__is_more_general(cl, c)):
                    cl["numeriosity"] += c["numeriosity"]
                    A.remove(c)
                    P.remove(c)
    
    def parameter_update(self, A: ActionSet, P: float, Pop: Population):
        n_sum = np.sum([c["numeriosity"] for c in A])
        
        for cl in A:
            cl["experience"] += 1
            
            if cl["experience"] < 1 / self.beta:
                cl["prediction"] = cl["prediction"] + (P - cl["prediction"]) / cl["experience"]
                cl["error"] = cl["error"] + (np.abs(P - cl["prediction"]) - cl["error"]) / cl["experience"]
                cl["act_size"] = cl["act_size"] + (n_sum - cl["act_size"]) / cl["experience"]
            else:
                cl["prediction"] = cl["prediction"] + self.beta * (P - cl["prediction"])
                cl["error"] = cl["error"] + self.beta * (np.abs(P - cl["prediction"]) - cl["error"])
                cl["act_size"] = cl["act_size"] + self.beta * (n_sum - cl["act_size"])
        self.__update_fitness(A)
        if(self.do_act_subsumption):
            self.__do_action_subsumptiono(A, Pop)
        
