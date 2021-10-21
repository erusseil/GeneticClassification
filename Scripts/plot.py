from gplearn.genetic import SymbolicRegressor
from sympy import *
import numpy as np
import pandas as pd
from gplearn.functions import make_function
from scipy.stats import norm
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from collections import OrderedDict


colors1 = ['blue','orange','salmon','teal','khaki','green','violet']
colors2 = ['darkblue','darkorange','tomato','darkslategrey','darkkhaki','darkgreen','darkviolet']


def scatter_plot(function,features,feature_toplot,dic_goals,classes = "Unclassified", thresh='None'):

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
        Default is "Unclassified" (all points will be labled "Unclassified")
        
    thresh: float
        Plot the threshold line. If 'None' no threshold will be plot
        Default is 'None'
    """
    
    #If we do not specify the classes, creates an array of zeros
    if type(classes) == str :
        if classes == "Unclassified":
            classes = np.array([classes]*len(features))
            
    #We order the dict by alphebetical order to match the colors :
    dic_goals = ({v: k for k, v in dic_goals.items()}) # Reverse the dict
    dic_goals = OrderedDict(sorted(dic_goals.items(), key=lambda t: t[0])) # Sort by numerical order
    dic_goals = ({v: k for k, v in dic_goals.items()}) # Reverse again
        
    
    # Plot all the points : one plt.scatter per type of points
    count = 0
    
    for i in sorted(np.unique(classes)):
        flag = classes == i
    
        if i == "Unclassified":
            color = 'gray'
        else :
            color = colors1[count]
        
        
        plt.scatter(features.loc[flag,feature_toplot],function.predict(features)[flag],label=i,color=color);
        count+=1
    
    
    # Plot all the goal lines
    for i in range(len(dic_goals)):
        line = list(dic_goals.keys())[i]
        name = dic_goals.get(line)
        
        plt.axhline(y=line, label='Target %s : %s'%(line,name),color=colors2[i])
    
    if thresh != 'None':
        plt.axhline(y=thresh, label='Treshold',color='black')
    
    
def histo_plot(function,features,dic_goals,bins,classes = "Unclassified", alpha = 0.7,plot_lines = True, thresh='None'):
   
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
        Default is "Unclassified" (all points will be labled "Unclassified")
        
    alpha: float
        Between 0 and 1. Defines the opacity of the plot.
        
    plot_lines: bool
        If True will plot the target values as vertical lines
        Default is True
        
    thresh: float
        Plot the threshold line. If 'None' no threshold will be plot
        Default is 'None'    
    """
    
    # If we do not specify the classes, creates an array of zeros
    if type(classes) == str :
        if classes == "Unclassified":
            classes = np.array([classes]*len(features))
   

    # We order the dict by alphebetical order to match the colors :
    dic_goals = ({v: k for k, v in dic_goals.items()}) # Reverse the dict
    dic_goals = OrderedDict(sorted(dic_goals.items(), key=lambda t: t[0])) # Sort by numerical order
    dic_goals = ({v: k for k, v in dic_goals.items()}) # Reverse again
    
    
    # Plot all the points : one plt.scatter per type of points
    count = 0
    for i in sorted(np.unique(classes)):
        flag = classes == i
        
        if i == "Unclassified":
            color = 'gray'
        else :
            color = colors1[count]
            
        plt.hist(function.predict(features)[flag],bins=bins,label=i,alpha=alpha,color=color);
        count+=1
    
    
    if plot_lines == True :
        # Plot all the goal lines
        for i in range(len(dic_goals)):
            line = list(dic_goals.keys())[i]
            name = dic_goals.get(line)

            plt.axvline(x=line, label='Target %s : %s'%(line,name),color=colors2[i])
        

    if thresh != 'None':
        plt.axvline(x=thresh, label='Threshold',color='black')