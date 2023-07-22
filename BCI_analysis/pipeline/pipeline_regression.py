import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from tqdm import tqdm

def linear_regression(F_aligned, DLC_aligned):

    print(f"Calculating regression fit for {F_aligned.shape[0]} neurons, May take time")
    n_features = DLC_aligned.shape[1]
    beta_vals = np.zeros((F_aligned.shape[0], n_features))
    scores = []
    intercept = []
    for neuron in tqdm(range(F_aligned.shape[0])):
        lr = LinearRegression()
        lr.fit(DLC_aligned.values, F_aligned[neuron])
        scores.append(lr.score(DLC_aligned.values, F_aligned[neuron]))
        beta_vals[neuron] = lr.coef_
        intercept.append(lr.intercept_)
    return scores, beta_vals, intercept

def ridge_regression(F_aligned, DLC_aligned):

    print(f"Calculating regression fit for {F_aligned.shape[0]} neurons, May take time")
    n_features = DLC_aligned.shape[1]
    beta_vals = np.zeros((F_aligned.shape[0], n_features))
    scores = []
    intercept = []
    for neuron in tqdm(range(F_aligned.shape[0])):
        lr = Ridge()
        lr.fit(DLC_aligned.values, F_aligned[neuron])
        scores.append(lr.score(DLC_aligned.values, F_aligned[neuron]))
        beta_vals[neuron] = lr.coef_
        intercept.append(lr.intercept_)
    return scores, beta_vals, intercept


# def linear_regression_cv(F_aligned, DLC_aligned,test_size = .1):
#     df_train, df_test = train_test_split(DLC_aligned, 
#                                          train_size = 1-test_size, 
#                                          test_size = test_size, 
#                                          random_state = 100)
    
#     print(f"Calculating regression fit for {F_aligned.shape[0]} neurons, May take time")
#     n_features = DLC_aligned.shape[1]
#     beta_vals = np.zeros((F_aligned.shape[0], n_features))
#     scores = []
#     intercept = []
#     for neuron in tqdm(range(F_aligned.shape[0])):
#         lr = LinearRegression()
#         lr.fit(DLC_aligned.values, F_aligned[neuron])
#         scores.append(lr.score(DLC_aligned.values, F_aligned[neuron]))
#         beta_vals[neuron] = lr.coef_
#         intercept.append(lr.intercept_)
#     return scores, beta_vals, intercept