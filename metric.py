import numpy as np
import pandas as pd

def metric(answers, user_csv):

    delta_c = np.abs(np.array(answers['C']) - np.array(user_csv['C']))
    hit_rate_c = np.int64(delta_c < 0.02)

    delta_t = np.abs(np.array(answers['TST']) - np.array(user_csv['TST']))
    hit_rate_t = np.int64(delta_t < 20)

    N = np.size(answers['C'])

    return np.sum(hit_rate_c + hit_rate_t) / 2 / N


def metricT(y_true, y_pred):
        delta_t = np.abs(np.array(y_true) - np.array(y_pred))
        hit_rate_t = np.int64(delta_t < 20)
        
        return hit_rate_t.mean()
    
    
def metricC(y_true, y_pred):
    delta_t = np.abs(np.array(y_true) - np.array(y_pred))
    hit_rate_t = np.int64(delta_t < 0.02)
        
    return hit_rate_t.mean()
