from gplearn.genetic import SymbolicRegressor
from sympy import *
import numpy as np
import pandas as pd
from gplearn.functions import make_function
from scipy.stats import norm
import pickle


converter = {
'sub': lambda x, y : x - y,
'div': lambda x, y : x/y,
'mul': lambda x, y : x*y,
'add': lambda x, y : x + y,
'neg': lambda x    : -x,
'pow': lambda x, y : x**y,
'sin': lambda x    : sin(x),
'cos': lambda x    : cos(x),
'inv': lambda x: 1/x,
'sqrt': lambda x: x**0.5,
'pow3': lambda x: x**3,
}



def evolve(features,classes,dic_classes,save,addfunc=[],population_size=1500,generations=300, stopping_criteria=0.01,
          p_crossover=0.6, p_subtree_mutation=0.2,p_hoist_mutation=0.05,p_point_mutation=0.1,max_samples=0.9,
           verbose=1,parsimony_coefficient=0.01,random_state=0): #add warm start
    
    """Generates a function that attracts specific classes towards inputed values
    
    Parameters
    ----------
    features: pd.DataFrame
        Data frame of each features used to describe the curves
         
    dic_classes: dict
        Dictionary associating a key class of objects to an attracting goal value
        
    save: str
        Name of the file containing the final function
        
    addfunc: list
        List of all additionnal operations to be added to the standard pool
        Default is []
        
    population_size: int
        Size of the population at each generation
        Default is 1500
        
    generations: int 
        Number of generations 
        Default is 300
    
    TO BE COMPLETED
    -----------------------------
    stopping_criteria: float
        Default is 0.01    
    p_crossover: float
        Default is 0.6
    p_subtree_mutation: float
        Default is 0.2
    p_hoist_mutation: float
        Default is 0.05
    p_point_mutation: float
        Default is 0.1
    max_samples: float
        Default is 0.9
    verbose: int
        Default is 1
    parsimony_coefficient: float
        Default is 0.01    
    -----------------------------
    
        
    alpha: float
        Between 0 and 1. Defines the opacity of the plot.
        
    """

    nb_obj = len(classes)
    
   # Create array of the goal values
    goals = np.zeros(nb_obj)
    for i in range(nb_obj):
        goals[i] = dic_classes.get(classes.iloc[i])
    
    # Define the set of possible operations
    function_set = ['add', 'sub', 'mul', 'div','cos','sin','neg','inv']
    if addfunc != []:
        function_set.append(addfunc)    
    
    gp = SymbolicRegressor(population_size=population_size,function_set=function_set,
                           generations=generations, stopping_criteria=stopping_criteria,
                           p_crossover=p_crossover, p_subtree_mutation=p_subtree_mutation,
                           p_hoist_mutation=p_hoist_mutation,
                           p_point_mutation=p_point_mutation,
                           max_samples=max_samples, verbose=verbose,
                           parsimony_coefficient=parsimony_coefficient,
                           random_state=random_state,feature_names=features.columns)
    
    gp.fit(features, goals)
    
    print('R2:',gp.score(features, goals))
    
    # Save the model
    with open('%s.pkl'%save, 'wb') as f:
        pickle.dump(gp, f)
        
        
    return gp