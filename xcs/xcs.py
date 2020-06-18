import sys, random, itertools, copy
import numpy as np

from abc import ABCMeta, abstractmethod

from xcs.rlcomponent import QlearnLikeRLComponent
from xcs.gacomponent import SimpleGAComponent
from xcs.population import Population
from xcs.matchset import MatchSet
from xcs.actionset import PredictArray, ActionSet
from xcs.environment import Environment
from xcs.classifier import Classifier

class XCS:
    def __init__(self, env: Environment, N=100, beta=0.1, alpha=0.1, eps_0=0.01, nu=5, 
                 gamma=0.71, theta_ga=25, chi=0.5, mu=0.01, theta_del=20,
                 delta=0.1, theta_sub=20, P_s=0.33, p_I=np.finfo(np.float32).eps,
                 e_I=np.finfo(np.float32).eps, f_I=np.finfo(np.float32).eps,
                 p_explr=0.5, theta_mna=1, do_ga_subsumption=False,
                 do_actionset_subsumption=False):
        self.N = N
        self.beta = beta
        self.alpha = alpha
        self.eps_0 = eps_0
        self.nu = nu
        self.gamma = gamma
        self.theta_ga = theta_ga
        self.chi = chi
        self.mu = mu
        self.theta_del = theta_del
        self.delta = delta
        self.theta_sub = theta_sub
        self.P_s = P_s
        self.p_explr = p_explr
        self.theta_mna = theta_mna
        self.do_ga_subsumption = do_ga_subsumption
        self.do_actionset_subsumption = do_actionset_subsumption
        Classifier.p_I = p_I
        Classifier.e_I = e_I
        Classifier.f_I = f_I
        
        self.env = env
        self.rp = QlearnLikeRLComponent(theta_mna, P_s, p_explr, alpha, beta, eps_0,
                                        nu, theta_sub, do_actionset_subsumption)
        self.ga = SimpleGAComponent(theta_ga, chi, mu, do_ga_subsumption)
        
        self.t = 0
        self.num_iter = 0
        self.max_iter = env.max_iter
        
        self.Pop = Population(N, len(env[0]), 1, theta_del, delta, empty=True)
        
    def run_experiment(self):
        before_rho = 0
        before_A = None
        before_sigma = None
        
        while True:
            sigma = self.env.get_situation(self.t)
            M = MatchSet(self.Pop, sigma, self.theta_mna, self.P_s, time=self.t)
            PA = PredictArray(M)
            act = PA.select_action(self.p_explr)
            A = ActionSet(M, act)
            rho = self.env.exec_action(self.t, act)
                        
            if(before_A is not None):
                P = before_rho + self.gamma * max(PA)
                self.rp.parameter_update(before_A, P, self.Pop)
                self.ga.run_evolve(before_A, before_sigma, self.Pop)
            
            if(self.env.is_end_problem()):
                P = rho
                self.rp.parameter_update(A, P, self.Pop)
                self.ga.run_evolve(A, sigma, self.Pop)
            else:
                before_A = copy.deepcopy(A)
                before_rho = rho
                before_sigma = sigma
            
            # print(f"{self.num_iter}: {len(self.Pop)}")
            print(f"========== Population: {len(self.Pop)} ==========")
            self.Pop.print()
            print(f"========== End {self.num_iter} ==========")

            self.t += 1
            self.num_iter += 1

            if self.num_iter > self.max_iter - 1:
                break