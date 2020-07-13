############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Local Search-3-opt

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-Local_Search-3-opt, File: Python-MH-Local Search-3-opt.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Local_Search-3-opt>

############################################################################

# Required Libraries
import pandas as pd
import random
import numpy  as np
import copy
from matplotlib import pyplot as plt 

# Function: Tour Distance
def distance_calc(Xdata, city_tour):
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m = k + 1
        distance = distance + Xdata[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

# Function: Euclidean Distance 
def euclidean_distance(x, y):       
    distance = 0
    for j in range(0, len(x)):
        distance = (x[j] - y[j])**2 + distance   
    return distance**(1/2) 

# Function: Initial Seed
def seed_function(Xdata):
    seed = [[],float("inf")]
    sequence = random.sample(list(range(1,Xdata.shape[0]+1)), Xdata.shape[0])
    sequence.append(sequence[0])
    seed[0] = sequence
    seed[1] = distance_calc(Xdata, seed)
    return seed

# Function: Build Distance Matrix
def build_distance_matrix(coordinates):
   a = coordinates
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

# Function: Tour Plot
def plot_tour_distance_matrix (Xdata, city_tour):
    m = np.copy(Xdata)
    for i in range(0, Xdata.shape[0]):
        for j in range(0, Xdata.shape[1]):
            m[i,j] = (1/2)*(Xdata[0,j]**2 + Xdata[i,0]**2 - Xdata[i,j]**2)    
    w, u = np.linalg.eig(np.matmul(m.T, m))
    s = (np.diag(np.sort(w)[::-1]))**(1/2) 
    coordinates = np.matmul(u, s**(1/2))
    coordinates = coordinates.real[:,0:2]
    xy = np.zeros((len(city_tour[0]), 2))
    for i in range(0, len(city_tour[0])):
        if (i < len(city_tour[0])):
            xy[i, 0] = coordinates[city_tour[0][i]-1, 0]
            xy[i, 1] = coordinates[city_tour[0][i]-1, 1]
        else:
            xy[i, 0] = coordinates[city_tour[0][0]-1, 0]
            xy[i, 1] = coordinates[city_tour[0][0]-1, 1]
    plt.plot(xy[:,0], xy[:,1], marker = 's', alpha = 1, markersize = 7, color = 'black')
    plt.plot(xy[0,0], xy[0,1], marker = 's', alpha = 1, markersize = 7, color = 'red')
    plt.plot(xy[1,0], xy[1,1], marker = 's', alpha = 1, markersize = 7, color = 'orange')
    return

# Function: Tour Plot
def plot_tour_coordinates (coordinates, city_tour):
    xy = np.zeros((len(city_tour[0]), 2))
    for i in range(0, len(city_tour[0])):
        if (i < len(city_tour[0])):
            xy[i, 0] = coordinates[city_tour[0][i]-1, 0]
            xy[i, 1] = coordinates[city_tour[0][i]-1, 1]
        else:
            xy[i, 0] = coordinates[city_tour[0][0]-1, 0]
            xy[i, 1] = coordinates[city_tour[0][0]-1, 1]
    plt.plot(xy[:,0], xy[:,1], marker = 's', alpha = 1, markersize = 7, color = 'black')
    plt.plot(xy[0,0], xy[0,1], marker = 's', alpha = 1, markersize = 7, color = 'red')
    plt.plot(xy[1,0], xy[1,1], marker = 's', alpha = 1, markersize = 7, color = 'orange')
    return

# Function: 2_opt
def local_search_2_opt(Xdata, city_tour):
    city_list = copy.deepcopy(city_tour)
    best_route = copy.deepcopy(city_list)
    seed = copy.deepcopy(city_list)        
    for i in range(0, len(city_list[0]) - 2):
        for j in range(i+1, len(city_list[0]) - 1):
            best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))           
            best_route[0][-1]  = best_route[0][0]                          
            best_route[1] = distance_calc(Xdata, best_route)    
            if (best_route[1] < city_list[1]):
                city_list[1] = copy.deepcopy(best_route[1])
                for n in range(0, len(city_list[0])): 
                    city_list[0][n] = best_route[0][n]          
            best_route = copy.deepcopy(seed) 
    return city_list

# Function: 3_opt
def local_search_3_opt(Xdata, city_tour, recursive_seeding = 1):
    if (recursive_seeding < 0):
        count = recursive_seeding - 1
    else:
        count = 0
    city_list = copy.deepcopy(city_tour)
    city_list_old = city_list[1]*2
    iteration = 0
    while (count < recursive_seeding):
        best_route   = copy.deepcopy(city_list)
        best_route_1 = local_search_2_opt(Xdata, best_route)
        best_route_2 = [[],1]
        best_route_3 = [[],1]
        best_route_4 = [[],1]
        best_route_5 = [[],1]       
        seed = copy.deepcopy(city_list)        
        for i in range(0, len(city_list[0]) - 3):
            for j in range(i+1, len(city_list[0]) - 2):
                for k in range(j+1, len(city_list[0]) - 1): 
                    best_route_2[0] = best_route[0][:i+1]+best_route[0][j+1:k+1]+best_route[0][i+1:j+1]+best_route[0][k+1:]
                    best_route_2[1] = distance_calc(Xdata, best_route_2)
                    best_route_3[0] = best_route[0][:i+1]+list(reversed(best_route[0][i+1:j+1]))+list(reversed(best_route[0][j+1:k+1]))+best_route[0][k+1:]
                    best_route_3[1] = distance_calc(Xdata, best_route_3)
                    best_route_4[0] = best_route[0][:i+1]+list(reversed(best_route[0][j+1:k+1]))+best_route[0][i+1:j+1]+best_route[0][k+1:]
                    best_route_4[1] = distance_calc(Xdata, best_route_4)
                    best_route_5[0] = best_route[0][:i+1]+best_route[0][j+1:k+1]+list(reversed(best_route[0][i+1:j+1]))+best_route[0][k+1:]
                    best_route_5[1] = distance_calc(Xdata, best_route_5)
                    
                    if(best_route_1[1]  < best_route[1]):
                        best_route = copy.deepcopy(best_route_1)                            
                    elif(best_route_2[1]  < best_route[1]):
                        best_route = copy.deepcopy(best_route_2)                            
                    elif(best_route_3[1]  < best_route[1]):
                        best_route = copy.deepcopy(best_route_3)                             
                    elif(best_route_4[1]  < best_route[1]):
                        best_route = copy.deepcopy(best_route_4)                            
                    elif(best_route_5[1]  < best_route[1]):
                        best_route = copy.deepcopy(best_route_5)
                            
                if (best_route[1] < city_list[1]):
                    city_list = copy.deepcopy(best_route)             
                best_route = copy.deepcopy(seed)
        count = count + 1  
        iteration = iteration + 1  
        print("Iteration = ", iteration, "-> Distance =", city_list[1])
        if (city_list_old > city_list[1] and recursive_seeding < 0):
             city_list_old = city_list[1]
             count = -2
             recursive_seeding = -1
        elif(city_list[1] >= city_list_old and recursive_seeding < 0):
            count = -1
            recursive_seeding = -2
    print(city_list)
    return city_list

######################## Part 1 - Usage ####################################

# Load File - A Distance Matrix (17 cities,  optimal = 1922.33)
X = pd.read_csv('Python-MH-Local Search-3-opt-Dataset-01.txt', sep = '\t') 
X = X.values

# Start a Random Seed
seed = seed_function(X)

# Call the Function
ls3opt = local_search_3_opt(X, city_tour = seed, recursive_seeding = -1)

# Plot Solution. Red Point = Initial city; Orange Point = Second City # The generated coordinates (2D projection) are aproximated, depending on the data, the optimum tour may present crosses
plot_tour_distance_matrix(X, ls3opt)

######################## Part 2 - Usage ####################################

# Load File - Coordinates (Berlin 52,  optimal = 7544.37)
Y = pd.read_csv('Python-MH-Local Search-3-opt-Dataset-02.txt', sep = '\t') 
Y = Y.values

# Build the Distance Matrix
X = build_distance_matrix(Y)

# Start a Random Seed
seed = seed_function(X)

# Call the Function
ls3opt = local_search_3_opt(X, city_tour = seed, recursive_seeding = -1)

# Plot Solution. Red Point = Initial city; Orange Point = Second City
plot_tour_coordinates(Y, ls3opt)
