import numpy as np
import pandas as pd


# pairwise difference between real-univariate covariate of treated VS control group
def pairDist(treated=np.array, control=np.array):
        
    D = treated[:, None] - control
    
    return D


# pairwise absolute difference between real-univariate covariates of treated VS control group
def abs_pairDist(treated=np.array, control=np.array):
        
    D = np.abs(treated[:, None] - control)
    
    return D


# pairwise difference between factor-valued (i.e. bounded integer-valued) covariates 
# (e.g. day of the week, month, ...) of treated VS control group, assuming the facotr levels are cyclic
# and only the shortest difference modulo nb_levels matters.
def pairModuloDist(treated=np.array, control=np.array, nb_levels=int):
    # test here
    
    categorical_treated = False
    t_str_value = []
    
    for i in treated:
        if isinstance(i, str):
            t_str_value.append(True)
            
    if np.any(t_str_value):
        categorical_treated = True
    
    if categorical_treated:
        treated = pd.get_dummies(treated, dummy_na=True)
        treated = treated.values.argmax(1)
        
    categorical_control = False
    c_str_value = []
    
    for i in control:
        if isinstance(i, str):
            c_str_value.append(True)
            
    if np.any(c_str_value):
        categorical_control = True
            
    if categorical_control:
        control = pd.get_dummies(control, dummy_na=True) # Add a column to indicate NaNs, if False NaNs are ignored.
        control = control.values.argmax(1) #Returns the indices of the maximum values along the y-axis.
        
    
    treated_control = pairDist(treated.astype(int), control.astype(int)) % nb_levels
    control_treated = pairDist(control.astype(int), treated.astype(int)) % nb_levels
    
    pmin = np.minimum(treated_control, np.transpose(control_treated))
    
    return pmin



# pairwise difference between covariates of treated VS control group
# Inputs: treated/control are of covariate vectors (one entry per unit, for a given covariate)
# Outputs: pairwise difference matrix
def pairdifference(treated=np.array, control=np.array):
    
    categorical = False
    str_value = []
    
    for i in treated:
        if isinstance(i, str):
            str_value.append(True)
            
    if np.any(str_value):
        categorical = True
            
    
    if categorical:
        
        nb_levels = len(set(treated))
        D_mod = pairModuloDist(treated, control, nb_levels)
        
        return D_mod
    
    else:
        
        D_abs = abs_pairDist(treated,control)
        
        return D_abs
    
    
def discrepancyMatrix(treated, control, thresholds, scaling=None):
    
    nb_covariates = treated.shape[1]
    
    nrow = treated.shape[0]
    ncol = control.shape[0]
    D = np.zeros(shape=(nrow, ncol))
    
    non_admissible = np.full((nrow, ncol), False)
    
    for i in range(0, nb_covariates):
        
        if not np.isnan(thresholds[i]):
            
            t = np.array(treated.iloc[:, i])
            c = np.array(control.iloc[:, i])
            
            differences = pairdifference(t, c)
            D = D + differences*scaling[i]
            
            
            differences[np.isnan(differences)] = 0
             
            if thresholds[i] >= 0:
            
                non_admissible = non_admissible + (differences > thresholds[i])
            
            elif thresholds[i] < 0:
            
                non_admissible = non_admissible + (differences <= np.abs(thresholds[i]))
    
    D = D / nb_covariates
    
    D[non_admissible] = np.nan
    
    return D