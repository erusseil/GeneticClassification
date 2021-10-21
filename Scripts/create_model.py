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
           verbose=1,parsimony_coefficient=0.01,random_state=42,metric='mean absolute error',tournament_size=20,
           n_jobs = 1): #add warm start
    
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
        
    population_size : integer, optional (default=1000)
        The number of programs in each generation.

    generations : integer, optional (default=20)
        The number of generations to evolve.

    tournament_size : integer, optional (default=20)
        The number of programs that will compete to become part of the next
        generation.

    stopping_criteria : float, optional (default=0.0)
        The required metric value required in order to stop evolution early.

        metric = str
            Function to be used to calculate the fitness of an equation : 'rmse' or 'mse' or default
            Default is ‘mean absolute error’
    
    p_crossover : float, optional (default=0.9)
        The probability of performing crossover on a tournament winner.
        Crossover takes the winner of a tournament and selects a random subtree
        from it to be replaced. A second tournament is performed to find a
        donor. The donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring in the next
        generation.

    p_subtree_mutation : float, optional (default=0.01)
        The probability of performing subtree mutation on a tournament winner.
        Subtree mutation takes the winner of a tournament and selects a random
        subtree from it to be replaced. A donor subtree is generated at random
        and this is inserted into the original parent to form an offspring in
        the next generation.

    p_hoist_mutation : float, optional (default=0.01)
        The probability of performing hoist mutation on a tournament winner.
        Hoist mutation takes the winner of a tournament and selects a random
        subtree from it. A random subtree of that subtree is then selected
        and this is 'hoisted' into the original subtrees location to form an
        offspring in the next generation. This method helps to control bloat.

    p_point_mutation : float, optional (default=0.01)
        The probability of performing point mutation on a tournament winner.
        Point mutation takes the winner of a tournament and selects random
        nodes from it to be replaced. Terminals are replaced by other terminals
        and functions are replaced by other functions that require the same
        number of arguments as the original node. The resulting tree forms an
        offspring in the next generation.

        Note : The above genetic operation probabilities must sum to less than
        one. The balance of probability is assigned to 'reproduction', where a
        tournament winner is cloned and enters the next generation unmodified.

    p_point_replace : float, optional (default=0.05)
        For point mutation only, the probability that any given node will be
        mutated.

    max_samples: float
        Default is 0.9
        
    verbose: int
        Default is 1
        
     parsimony_coefficient : float or "auto", optional (default=0.001)
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.
        If "auto" the parsimony coefficient is recalculated for each generation
        using c = Cov(l,f)/Var( l), where Cov(l,f) is the covariance between
        program size l and program fitness f in the population, and Var(l) is
        the variance of program sizes.
        
    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for `fit`. If -1, then the number
        of jobs is set to the number of cores.
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
                           p_hoist_mutation=p_hoist_mutation,metric=metric,
                           p_point_mutation=p_point_mutation,
                           max_samples=max_samples, verbose=verbose,
                           parsimony_coefficient=parsimony_coefficient,
                           random_state=random_state,feature_names=features.columns,
                           tournament_size=tournament_size, n_jobs=n_jobs)
    
    gp.fit(features, goals)
    
    print('R2:',gp.score(features, goals))
    
    # Save the model
    with open('%s.pkl'%save, 'wb') as f:
        pickle.dump(gp, f)
        
        
    return gp