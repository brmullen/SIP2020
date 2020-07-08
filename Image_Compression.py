#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 18:39:27 2020

@author: saathvikdirisala
"""

import matplotlib.image as img
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from time import sleep

rd = img.imread('MBJ.jpg')
image = np.mean(rd,-1)

U, S, VT = np.linalg.svd(image, full_matrices = False)
S = np.diag(S)

j = 0
for r in [50,100,400]:
    Xapprox = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
    plt.figure(j+1)
    j+=1
    img = plt.imshow(Xapprox)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r=' + str(r))
    plt.show()
    sleep(1)

'''plt.figure(1)
plt.semilogy(np.diag(S))
plt.title('Singular Values')
plt.show()'''

plt.figure(2)
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.title('Singular Values: Cumulative Sum')
plt.show()