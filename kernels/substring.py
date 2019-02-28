# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:37:44 2019

@author: salma
"""

import numpy as np
from kernels.default import Kernel

__all__ = ['SubstringKernel']


class SubstringKernel(Kernel):
    """
    Kernel described in slide 348 of http://members.cbio.mines-paristech.fr/~jvert/svn/kernelcourse/slides/master2017/master2017.pdf
    
    and the paper by Lohdi et al. 2002 found in http://www.jmlr.org/papers/volume2/lodhi02a/lodhi02a.pdf
    
    partially inspired from https://github.com/timshenkao/StringKernelSVM/blob/master/stringSVM.py
    """

    def __init__(self, length=3, decay_param=0.05):
        super(SubstringKernel, self).__init__()
        self.length = length
        self.decay_param = decay_param
        self.MEMOIZER_AUX = {}
        
    def embed(self, sequences):
        return sequences
    
    def K(self, s, t, n):
        print('K')
        if min(len(s), len(t)) < n:
            return 0
        partial_sum = 0
        for j in range(1, len(t)):
            # sum for t_j = x
            if t[j] == s[-1]:
                partial_sum += self.B(s[:-1], t[:j], n-1)
        return self.K(s[:-1], t, n) + partial_sum * self.decay_param**2
    
    def B(self, s, t, n):
        """
        Auxiliary function (K' in the paper)
        """
        print('B')
        if n == 0:
            return 1
        elif min(len(s), len(t)) < n:
            return 0
        else:
            partial_sum = 0
            for j in range(1, len(t)):
                if t[j] == s[-1]:
                    partial_sum += self.B(s[:-1], t[:j], n-1) * (self.decay_param ** len(t) - (j + 1) + 2)
            return partial_sum + self.decay_param * self.B(s[:-1], t, n)
        

    def apply(self, s, t):
        if s == t:
            return 1
        if (s, s) in self.MEMOIZER_AUX.keys():
            K_ss = self.MEMOIZER_AUX[(s, s)]
        else:
            K_ss = self.K(s, s, self.length)
        if (t, t) in self.MEMOIZER_AUX.keys():
            K_tt = self.MEMOIZER_AUX[(t, t)]
        else:
            K_tt = self.K(t, t, self.length)
        K_st = self.K(s, t, self.length)
        return K_st / np.sqrt(K_ss*K_tt)
        