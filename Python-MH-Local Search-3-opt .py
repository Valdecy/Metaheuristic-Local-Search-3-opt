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
import copy

# Function: Distance
def distance_calc(Xdata, route):
    distance = 0
    for k in range(0, len(route[0])-1):
        m = k + 1
        distance = distance + Xdata.iloc[route[0][k]-1, route[0][m]-1]            
    return distance

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
    count = 0
    city_list = copy.deepcopy(city_tour)
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
                        best_route[1] = copy.deepcopy(best_route_1[1])
                        for n in range(0, len(best_route[0])): 
                            best_route[0][n] = best_route_1[0][n]
                            
                    elif(best_route_2[1]  < best_route[1]):
                        best_route[1] = copy.deepcopy(best_route_2[1])
                        for n in range(0, len(best_route[0])): 
                            best_route[0][n] = best_route_2[0][n] 
                            
                    elif(best_route_3[1]  < best_route[1]):
                        best_route[1] = copy.deepcopy(best_route_3[1])
                        for n in range(0, len(best_route[0])): 
                            best_route[0][n] = best_route_3[0][n] 
                            
                    elif(best_route_4[1]  < best_route[1]):
                        best_route[1] = copy.deepcopy(best_route_4[1])
                        for n in range(0, len(best_route[0])): 
                            best_route[0][n] = best_route_4[0][n]
                            
                    elif(best_route_5[1]  < best_route[1]):
                        best_route[1] = copy.deepcopy(best_route_5[1])
                        for n in range(0, len(best_route[0])): 
                            best_route[0][n] = best_route_5[0][n] 
                            
                if (best_route[1] < city_list[1]):
                    city_list[1] = copy.deepcopy(best_route[1])
                    for n in range(0, len(city_list[0])): 
                        city_list[0][n] = best_route[0][n]              
                best_route = copy.deepcopy(seed)
        count = count + 1  
        print("Iteration = ", count, "->", city_list)
    return city_list

######################## Part 1 - Usage ####################################

X = pd.read_csv('Python-MH-Local Search-3-opt-Dataset-01.txt', sep = '\t') #17 cities => Optimum = 2085

cities = [[   1,  2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,   1   ], 4722]
ls3opt = local_search_3_opt(X, city_tour = cities, recursive_seeding = 5)
