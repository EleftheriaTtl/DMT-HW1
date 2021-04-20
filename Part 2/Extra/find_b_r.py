#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 16:02:42 2021

@author: psh
"""

import numpy as np
from scipy.optimize import fsolve, root_scalar

j = 0.95
n = 300
p = 0.97

def myFunction(b):
   return 1-(1-j**(n/b))**b - p

z = root_scalar(myFunction, bracket=[1, 300])
z = z.root

# Find closeset integer for which function is greater than 0
if myFunction(np.ceil(z)) > 0:
    b = np.ceil(z)
else:
    b = np.floor(z)

print(np.ceil(z), myFunction(np.ceil(z)))
print(np.floor(z), myFunction(np.floor(z)))

r = n / b
# We choose 12 since value of function in 12 is greater than 0. Also 300 is divisible by 12
# so 300/12 = 25=r,  b=12

print(r, b)
