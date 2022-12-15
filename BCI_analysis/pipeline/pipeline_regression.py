import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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
