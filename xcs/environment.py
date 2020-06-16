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
    def __init__(self, N_addr, max_iter=10000):
        self.k = N_addr
        self.N = self.k + 2 ** self.k
        self.max_iter = max_iter
        self.bit_array = np.random.randint(0, 2, (self.max_iter, self.N), dtype=bool)
        
    def get_situation(self, t):
        return self.bit_array[t, :]
        
    def exec_action(self, t, act, reward=1000):
        idx_true = self.__bits_to_int(self.bit_array[t, 0:self.k])
        res_true = self.bit_array[t, self.k + idx_true]
        
        ret = reward if act == res_true else 0
            
        return ret
    
    def __iter__(self):
        self.__idx_current = 0
        return self
    
    def __next__(self):
        if self.__idx_current == len(self):
            raise StopIteration()
            
        idx = self.__idx_current
        self.__idx_current += 1
            
        return self.bit_array[idx]
    
    def __getitem__(self, idx):
        return self.bit_array[idx]
    
    def __len__(self):
        return len(self.bit_array)
    
    def __bits_to_int(self, bits_list):
        def mypackbits(X, reverse=True):
            p = np.power(2, np.arange(X.shape[-1]))
            if reverse:
                p = p[::-1]
            return np.dot(X, p)

        idx = mypackbits(bits_list)
        
        return idx