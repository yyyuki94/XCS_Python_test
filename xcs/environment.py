import sys, random, itertools, copy
import numpy as np

from abc import ABCMeta, abstractmethod

# 環境の抽象クラス
class Environment(metaclass=ABCMeta):
    @abstractmethod
    def get_situation(self, t):
        pass
    
    @abstractmethod
    def exec_action(self, act, reward):
        pass

    
# マルチプレクサ問題の環境クラス
class MuxProblemEnvironment(Environment):
    def __init__(self, N_addr, max_iter=1000):
        self.k = N_addr
        self.N = self.k + 2 ** self.k
        self.max_iter = max_iter
        self.data = np.loadtxt(f"data/Mux-{self.N}.csv", delimiter=",")
        self.X = self.data[:, :-1]
        self.y = self.data[:, -1]
        self.time_table = np.random.choice(range(len(self.data)), max_iter)
        
    def get_situation(self, t):
        return self.X[self.time_table[t], :]
        
    def exec_action(self, t, act, reward=1000):
        true_val = self.y[self.time_table[t]]
        ret = reward if act == true_val else 0
            
        return ret
    
    def __iter__(self):
        self.__idx_current = 0
        return self
    
    def __next__(self):
        if self.__idx_current == len(self):
            raise StopIteration()
            
        idx = self.__idx_current
        self.__idx_current += 1
            
        return self.data[self.time_table[idx]]
    
    def __getitem__(self, idx):
        return self.data[self.time_table[idx]]
    
    def __len__(self):
        return len(self.time_table)
    
    def __bits_to_int(self, bits_list):
        def mypackbits(X, reverse=True):
            p = np.power(2, np.arange(X.shape[-1]))
            if reverse:
                p = p[::-1]
            return np.dot(X, p)

        idx = mypackbits(bits_list)
        
        return idx