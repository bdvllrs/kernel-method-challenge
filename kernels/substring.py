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
    
    """

    def __init__(self, memoize_conf, length=3, decay_param=0.5):
        super(SubstringKernel, self).__init__(memoize_conf)
        self.length = length
        self.decay_param = decay_param
        if 'K_xx' in self.memoizer:
            self.K_xx = self.memoizer['K_xx']
        else:
            self.K_xx = {}
        if 'B_st' in self.memoizer:
            self.B_st = self.memoizer['B_st']
        else:
            self.B_st = {}
        
    def embed(self, sequences):
        return sequences
    
    def K(self, s, t, n):
    
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

        if n == 0:
            return 1
        elif min(len(s), len(t)) < n:
            return 0
        if (s[:-1], t, n) in self.B_st.keys():
            B1 = self.B_st[(s[:-1], t, n)]
        else:
            B1 = self.B(s[:-1], t, n)
            #self.B_st[(s[:-1], t, n)] = B1
        if (s, t[:-1], n) in self.B_st.keys():
            B2 = self.B_st[(s, t[:-1], n)]
        else:
            B2 = self.B(s, t[:-1], n)
            #self.B_st[(s, t[:-1], n)] = B2 
            
        if (s[:-1], t[:-1], n) in self.B_st.keys():
            B3 = self.B_st[(s[:-1], t[:-1], n)]
        else:
            B3 = self.B(s[:-1], t[:-1], n)
            #self.B_st[(s[:-1], t[:-1], n)] = B3
        result = self.decay_param * (B1 + B2 - self.decay_param * B3)
        if s[-1] == t[-1]:
            if (s[:-1], t[:-1], n-1) in self.B_st.keys():
                B4 = self.B_st[(s[:-1], t[:-1], n-1)]
            else:
                B4 = self.B(s[:-1], t[:-1], n-1)
                #self.B_st[(s[:-1], t[:-1], n-1)] = B4
            result += (self.decay_param ** 2) * B4
        return result
                

    def apply(self, s, t):
        if s == t:
            return 1
        if (s, s) in self.K_xx.keys():
            K_ss = self.K_xx[(s, s)]
        else:
            K_ss = self.K(s, s, self.length)
            self.K_xx[(s,s)] = K_ss
        if (t, t) in self.K_xx.keys():
            K_tt = self.K_xx[(t, t)]
        else:
            K_tt = self.K(t, t, self.length)
            self.K_xx[(t,t)] = K_tt
        K_st = self.K(s, t, self.length)
        return K_st / np.sqrt(K_ss*K_tt)
        
