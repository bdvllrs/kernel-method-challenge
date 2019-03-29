# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:27:29 2019

@author: salma
"""

import numpy as np
from kernels.default import Kernel
from classifiers.HMM import HMM_EM

__all__ = ['FisherHMMKernel']


class FisherHMMKernel(Kernel):
    """
    
    """
    def __init__(self, n_states=4):
        super(FisherHMMKernel, self).__init__()
        self.n_states = n_states
        
        
    def fit_model(self, sequence):
        hmm = HMM_EM(n_states=self.n_states)
        hmm.fit(sequence)
        
        
        
    def embed(self, sequences):
        return sequences
    


