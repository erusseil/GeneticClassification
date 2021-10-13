from gplearn.genetic import SymbolicRegressor
from sympy import *
import numpy as np
import pandas as pd
from gplearn.functions import make_function
from scipy.stats import norm
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


colors1 = ['blue','orange','salmon','teal','khaki','green','violet']
colors2 = ['darkblue','darkorange','tomato','darkslategrey','darkkhaki','darkgreen','darkviolet']

dic_classes = {90:'SNIa',67:'SNIa-91bg',
             52:'SNIax',42:'SNII',
             62:'SNIbc',95:'SLSN-I',
             15:'TDE',64:'KN',
             88:'AGN',92:'RRL',
             65:'M-dwarf',16:'EB',
             53:'Mira',6:'Lens-Single',
             994:'PISN', 0:'UNKNOWN'}

def scatter_plot(function,features,feature_toplot,dic_goals,classes = 0):

    """Scatter plots : result of the function versus one of the features
    
    Parameters
    ----------
    function: gplearn.genetic.SymbolicRegressor
        Result of the function obtain with 'evolution' file
        
    features: pd.DataFrame
        Data frame of each feature used to build the function.
        
    feature_toplot: str
        Name of the feature to use for the 'x' axis
     
    dic_goals: dict
        Dictionary associating a key goal value to a class
        
    classes: np.array
        Array of all the classes in case they are known in advance.
        Default is 0 (all points will be labled 'unknown')
        
    """
    
    #If we do not specify the classes, creates an array of zeros
    if type(classes) == int :
        classes = np.zeros(len(features))
    
    # Plot all the points : one plt.scatter per type of points
    count = 0
    for i in np.unique(classes):
        flag = classes == i
    
        if i == 0:
            color = 'gray'
        else :
            color = colors1[count]

        plt.scatter(features.loc[flag,feature_toplot],function.predict(features)[flag],label=dic_classes.get(i),color=color);
        count+=1
    
    
    # Plot all the goal lines
    for i in range(len(dic_goals)):
        line = list(dic_goals.keys())[i]
        name = dic_goals.get(line)
        
        plt.axhline(y=line, label='Target %s : %s'%(line,name),color=colors2[i])
    
    
    
    
    
    
    
def histo_plot(function,features,dic_goals,bins,classes = 0, alpha = 0.7):
   
    """Histogram plots : distribution of the function results
    
    Parameters
    ----------
    function: gplearn.genetic.SymbolicRegressor
        Result of the function obtain with 'evolution' file
        
    features: pd.DataFrame
        Data frame of each feature used to build the function.
        
     
    dic_goals: dict
        Dictionary associating a key goal value to a class
        
    classes: np.array
        Array of all the classes in case they are known in advance.
        Default is 0 (all points will be labled 'unknown')
        
    alpha: float
        Between 0 and 1. Defines the opacity of the plot.
        
    """
    
    #If we do not specify the classes, creates an array of zeros
    if type(classes) == int :
        classes = np.zeros(len(features))
   

    # Plot all the points : one plt.scatter per type of points
    count = 0
    for i in np.unique(classes):
        flag = classes == i
        
        if i == 0:
            color = 'gray'
        else :
            color = colors1[count]
            
        plt.hist(function.predict(features)[flag],bins=bins,label=dic_classes.get(i),alpha=alpha,color=color);
        count+=1
    
    
    # Plot all the goal lines
    for i in range(len(dic_goals)):
        line = list(dic_goals.keys())[i]
        name = dic_goals.get(line)
        
        plt.axvline(x=line, label='Target %s : %s'%(line,name),color=colors2[i])
        

  