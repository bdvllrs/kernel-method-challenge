# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:43:46 2019

@author: salma
"""
import numpy as np
from scipy.stats import multivariate_normal as mvn

def one_hot_encode_letter(letter):
    if letter =='A':
        return [1,0,0,0]
    elif letter =='C':
        return [0,1,0,0]
    elif letter=='G':
        return [0,0,1,0]
    elif letter =='T':
        return [0,0,0,1]
    
def one_hot_encode_sequence(sequence):
    vector = [one_hot_encode_letter(letter) for letter in sequence]
    return np.array(vector)

def compute_logsum(L):
    max_idx = np.argmax(L)
    max_l = L[max_idx]
    temp = [L[i]-max_l for i in range(len(L)) if i!=max_idx]
    return  max_l + np.log(1+np.sum(np.exp(temp)))


def alpha_recursion(X, log_emission_probas, log_transition_probas, pi0):
    T, K = log_emission_probas.shape
    log_alphas = np.zeros((T, K))
    log_alphas[0,:] = np.log(pi0+1e-14) + log_emission_probas[0,:]
    for t in range(T-1):
      for zt1 in range(K):
        L = log_transition_probas[zt1, :] + log_alphas[t, :] 
        log_sum = compute_logsum(L)
        log_alphas[t+1, zt1] = log_emission_probas[t+1, zt1] + log_sum 
    return log_alphas
  
    
def beta_recursion(X, log_emission_probas, log_transition_probas):
    T, K = log_emission_probas.shape
    log_betas = np.zeros((T,K))
    log_betas[T-1,:] = np.zeros((1,K))
    for t in range(T-2, -1, -1):
      for zt in range(K):
        L = log_transition_probas[:, zt] + log_betas[t+1,:] + log_emission_probas[t+1, :]
        log_sum = compute_logsum(L)
        log_betas[t, zt] = log_sum
    return log_betas
  
    
def estimate_state_probas(log_alphas, log_betas):
    T, K = log_alphas.shape
    log_state_probas = np.zeros((T, K))
    for t in range(T):
        L = log_alphas[t,:] + log_betas[t,:]
        log_sum = compute_logsum(L)
        log_state_probas[t,:] = log_alphas[t,:] + log_betas[t,:] - log_sum
    return log_state_probas
  
  
def estimate_joint_state_probas(X, log_emission_probas, log_transition_probas, log_betas, log_state_probas):
    T, K = log_state_probas.shape
    log_joint_probas = np.zeros((T, K, K))
    for t in range(T-1):
        for state in range(K):
            for next_state in range(K):
                 log_joint_probas[t, state, next_state] = log_state_probas[t, state] \
                                                        + log_transition_probas[next_state, state] \
                                                        + log_emission_probas[t+1, next_state] \
                                                        + log_betas[t+1, next_state] \
                                                        - log_betas[t, state]
    return log_joint_probas

class HMM_EM:
    def __init__(self, n_states=4, n_observations=4, max_iter=1000, epsilon=1e-6):
        self.n_states = n_states
        self.n_observations = n_observations
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.pi0 = np.ones((n_states,))/n_states
        A = np.random.rand(n_states, n_states)
        self.transition_matrix = A/A.sum(axis=1)[:,None]
        self.emission_matrix = np.ones((n_states, n_observations))/n_states
        
    def compute_probas(self, X, pi0, log_transition_probas, Q):
        log_emission_probas = np.log(np.dot(X,Q.T)+1e-14)
        log_alphas = alpha_recursion(X, log_emission_probas, log_transition_probas, pi0)
        log_betas =  beta_recursion(X, log_emission_probas, log_transition_probas)
        log_state_probas = estimate_state_probas(log_alphas, log_betas)
        log_joint_probas = estimate_joint_state_probas(X, log_emission_probas, log_transition_probas, log_betas, log_state_probas)
        emission_probas = np.exp(log_emission_probas)
        state_probas = np.exp(log_state_probas)
        joint_probas = np.exp(log_joint_probas)
        return (emission_probas, state_probas, joint_probas)
      
    def fit(self, X):
        self.X_train = X
        Q = self.emission_matrix
        A = self.transition_matrix
        pi0 = self.pi0
        k = self.n_states
        l = self.n_observations
        i = 0
        log_transition_probas = np.log(A+1e-14)
        log_likelihood_old = self.compute_log_likelihood(X)
        converged = False
        while i < self.max_iter and not converged:
            # E-step
            probas = self.compute_probas(X, pi0, log_transition_probas, Q)
            emission_probas, state_probas, joint_probas = probas
            # M-step
            pi0 = state_probas[0,:]
            log_transition_probas =  [[compute_logsum(np.log(joint_probas[:-1, i, j]+1e-14)) - 
                                       compute_logsum(np.log(state_probas[:-1, j]+1e-14)) for i in range(k)] for j in range(k)]
            log_transition_probas = np.array(log_transition_probas)
            Q = [[np.sum(state_probas[:, j]*X[:,i])/np.sum(state_probas[:, j]) for i in range(l)] for j in range(k)]
            Q = np.array(Q)
            A = np.exp(log_transition_probas)
            log_likelihood = self.compute_log_likelihood(X, probas, Q, A, pi0)
            print(log_likelihood)
            if (abs((log_likelihood - log_likelihood_old)/log_likelihood_old)) < self.epsilon:
                converged = True
                print('EM algorithm converged after {} iterations. Final log likelihood : {}'.format(i+1, log_likelihood))
            log_likelihood_old = log_likelihood
            i+=1
        self.pi0 = pi0
        self.emission_matrix = Q
        self.transition_matrix = A
        
        
    def viterbi(self, X=None):
        if X is None:
            X = self.X_train
        emission_matrix = np.array([mvn.pdf(X, self.means[k], self.covariances[k]) for k in range(self.n_states)]).T
        T, K = np.shape(emission_matrix)
        path = np.zeros_like(emission_matrix)
        log_tm = np.log(self.transition_matrix)
        log_em = np.log(emission_matrix)
        path_p = np.zeros_like(emission_matrix)
        path_p[0, :] = np.log(self.pi0*emission_matrix[0, :])
        for t in range(1, T):
            for k in range(K):
                prev_p = path_p[t-1, :]
                path[t, k] = np.argmax(prev_p + log_tm[k,:] + log_em[t, k])
                path_p[t, k] = np.max(prev_p + log_tm[k,:] + log_em[t, k])
        best_path = np.zeros(T)
        best_path[-1] = np.argmax(path_p[t, :])
        for t in range(T-1, 1, -1):
            best_path[t-1] = path[t, int(best_path[t])]
        return best_path
      
    def compute_log_likelihood(self, X, probas=None, Q = None, A = None, pi0 = None):
        if Q is None:
            Q = self.emission_matrix
        if A is None:
            A = self.transition_matrix
        if pi0 is None:
            pi0 = self.pi0
        if probas is None:
            probas = self.compute_probas(X, pi0, np.log(A+1e-14), Q)
        emission_probas, state_probas, joint_probas = probas
        T, K = state_probas.shape
        log_likelihood = np.nansum([[[joint_probas[t, i, j]*np.log(A[j, i]+1e-14) for i in range(K)] for j in range(K)] for t in range(T-1)])
        log_likelihood += np.nansum([state_probas[0, i]*np.log(pi0[i]+1e-14) for i in range(K)]) # used nansum to avoid problems when pi0(state)=0, in that case we have 0log(0) which is approx 0
        log_likelihood += np.nansum([[state_probas[t, i]*np.log(emission_probas[t,i]+1e-14) for i in range(K)] for t in range(T)])
        return log_likelihood
                  
            