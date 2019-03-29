# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:07:37 2019

@author: salma
"""
import numpy as np

def projected_gradient(xn, grad, projection, n_iter=50000, tol=1e-6):
    gamma = 0.1
    delta = 1.5
    lbd = delta / 2
    oldx = xn
    for k in range(n_iter):
        yn = xn - gamma * grad(xn)
        xn = xn + lbd * (projection(yn) - xn)
        if np.linalg.norm(oldx - xn) <= tol:
            print("Proj-Grad - Finished after", k + 1, "iterations.")
            break
        oldx = xn
    return xn