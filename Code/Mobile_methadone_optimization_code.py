# Built 12-07-22 by AB
# Solves mobile problem nationwide

# Import packages
import pandas as pd
import time
import csv
import os
import folium
import gurobipy as gp
from gurobipy import GRB
#from itertools import combinations
from itertools import permutations
from numpy import nan
import numpy as np

# Fixed Parameters
H = 7*60*60 #Hours in day.  Allows for 1-hour setup time (assumed no tear-down time)
max_dist = 1.5*60*60 #1.5 hour maximum travel from clinic to pharmacies
eps = 0.01 #Penalty term per edge used

#Parameters to search over
setup_time_list = [10*60, 15*60, 30*60, 60*60] #10, 15, 30 and 60 minute setup
serve_time_list = [1*60, 2.5*60, 5*60, 10*60] #1, 2.5, 5, and 10 minutes per customer
max_travel_list = [10*60, 25*60] #10 and 25 minute maximum travel for clients to mobile clinic


#Write initial dataframe
Results_df = pd.DataFrame(columns = ["State", "Starting Clinic", "Setup Param", "Service Param", "Max Travel Param", "Pharmacy Stops", "Demand Served", "Drive Time", "Service Time", "Setup Time", "Computational Run Time"])
Results_df.to_csv(path_or_buf="/home/bonifontea/Mobile_Methadone/Results/Optimization_12_07_22_Nationwide/Aggregate_results.csv",
									sep=",",
									index = False)


# Load Clinic and Tracts data - consistent across States
clinic_data = pd.read_csv('/home/bonifontea/Mobile_Methadone/Data/Nationwide/1 - Methadone_Clinics_2020.csv', index_col='ClinicID', encoding='latin-1')
tract_data = pd.read_csv('/home/bonifontea/Mobile_Methadone/Data/Nationwide/3 - Full_tract_data.csv', encoding='latin-1')
States = tract_data.State.unique()
States = np.delete(States, np.where((States == 'Connecticut') | (States == 'Delaware') | (States =='Rhode Island'))) #Remove states with no unserved demand

#########################
## Component functions ##
#########################
## Calculate shortest subtour for subtour elimination constraints
def subtour(edges):
	nodes = set(i for e in edges for i in e)
	unvisited = list(nodes)
	cycle = list(nodes)
	while unvisited:  # true if list is non-empty
		thiscycle = []
		neighbors = unvisited
		while neighbors:
			current = neighbors[0]
			thiscycle.append(current)
			unvisited.remove(current)
			neighbors = [j for i, j in edges.select(current, '*')
						 if j in unvisited]
		if len(thiscycle) <= len(cycle):
			cycle = thiscycle # New shortest subtour
	return cycle

## Subtour elimination function
def subtourelim(model, where):
	global subtour_iterations
	if where == GRB.Callback.MIPSOL:
		# make a list of edges selected in the solution
		vals = model.cbGetSolution(model._xij)
		selected = gp.tuplelist((i, j) for i, j in model._xij.keys()
							 if vals[i, j] > 0.5)
		tour = subtour(selected)
		if len(tour) < len(selected): #len(selected) is total number of edges	
			model.cbLazy(gp.quicksum(model._xij[i, j] for i, j in permutations(tour, 2))
						 <= len(tour)-1)						


###############
## Runs code ##
###############


for state in States:
#for state in ['Alabama','Arizona','Arkansas']:
    tic = time.perf_counter()
    
    ################
    ## Loads data ##
    ################
    state_data = tract_data[tract_data.State == state]
    state_data = state_data[state_data.Tract_Population > 0] #Restrict to only those with positive population
    
    state_clinics = clinic_data[clinic_data.State == state]
    
    state_pharmacies = pd.read_csv('/home/bonifontea/Mobile_Methadone/Data/Nationwide/By State/' + state + '/2 - ' + state + '_Pharmacies.csv', index_col='Cluster', encoding='latin-1')

    # Load drive distances
    cp_distances_data = pd.read_csv('/home/bonifontea/Mobile_Methadone/Data/Nationwide/By State/' + state + '/4 - ' + state + '_Pharmacy_Clinic_Pairs.csv', index_col='ID', encoding='latin-1')
    pp_distances_data = pd.read_csv('/home/bonifontea/Mobile_Methadone/Data/Nationwide/By State/' + state + '/5 - ' + state + '_Pairwise_Pharmacies.csv', index_col='ID', encoding='latin-1')
    # Combine together
    distances_data = pd.concat([cp_distances_data, pp_distances_data])
    # Create dictionary from distances
    distances_dict = {tuple(x[:2]):x[2] for x in distances_data[['Origin', 'Destination', 'Drive_Time']].values}

    # Load tracts served by each cluster (don't need distances).  Includes demand by tract
    tract_cluster_data = pd.read_csv('/home/bonifontea/Mobile_Methadone/Data/Nationwide/By State/' + state + '/6 - ' + state + '_Tracts_served_by_Pharmacies.csv', encoding='latin-1')


    # Define sets
    clinics = list(state_clinics.index)
    pharmacies = list(state_pharmacies.index)
    tracts =  state_data['GEOID'].unique().tolist()


    # Gurobi multidict to capture variables
    edges, drive_times = gp.multidict(distances_dict)



    ##################################
    ## Loop over sensitivity values ##
    ##################################

    for setup_time in setup_time_list:
        for serve_time in serve_time_list:
            for max_travel in max_travel_list:
                
                #Reduce for maximum allowable travel time
                this_tract_cluster_data = tract_cluster_data[tract_cluster_data["Drive_Time"] <= max_travel]
                tract_cluster_dict = {tuple(x[:2]):x[2] for x in this_tract_cluster_data[['GEOID', 'Cluster', 'Tract_methadone_unserved']].values}
                print(tract_cluster_dict) #For debugging
                
                customers, demand_served = gp.multidict(tract_cluster_dict)

                ########################
                ## Defines variables  ##
                ########################
                MMmodel = gp.Model("Mobile Methadone")

                # yp - binary if we stop at pharmacy p
                yp = MMmodel.addVars(pharmacies, vtype=GRB.BINARY, name="yp")

                # xij - binary if we travel from location i to location j (pharmacy or clinic)
                xij = MMmodel.addVars(edges, vtype=GRB.BINARY, name="xij")

                # zij - if we serve customer i at pharmacy j.  binary assumes full demand must be served, continuous allows fractional demand to be served
                #zij = MMmodel.addVars(customers, vtype=GRB.BINARY, name="zij")
                zij = MMmodel.addVars(customers, vtype=GRB.CONTINUOUS, name="zij")


                ##########################
                ## Creates constraints	##
                ##########################

                ### Time constraint
                MMmodel.addConstr(xij.prod(drive_times) + serve_time*zij.prod(demand_served) + setup_time*yp.sum() <= H, name="Time constraint")


                ### Routing constraints

                # If we make a stop, have to travel to and from it
                MMmodel.addConstrs((xij.sum('*',pharm) == yp[pharm] for pharm in yp), name="Travel to pharmacy") 
                MMmodel.addConstrs((xij.sum(pharm,'*') == yp[pharm] for pharm in yp), name="Travel from pharmacy")

                # Must leave one clinic
                #Guaranteed to be overwritten
                start_loc_1_const = MMmodel.addConstr(1 == 1, name="Selected clinic start") 
                start_loc_2_const = MMmodel.addConstr(1 == 1, name="Selected clinic start") 


                # Can only use one starting clinic and must end in clinic (dummy)
                #MMmodel.addConstr(gp.quicksum(xij.sum(clinic,'*') for clinic in clinics) == 1, name="Limited clinic start")
                #MMmodel.addConstr(gp.quicksum(xij.sum('*',clinic) for clinic in clinics)  == 1, name="No clinic return")
                MMmodel.addConstr(gp.quicksum(xij.sum(clinic,'*') for clinic in clinics)+gp.quicksum(xij.sum('*',clinic) for clinic in clinics)	 == 2, name="Limited clinic start")

                #### Service constraints
                #Can only serve demand if we stop a tract
                MMmodel.addConstrs((zij[tract, pharm] <= yp[pharm] for (tract,pharm) in zij), name="Serve if pharmacy open") 

                #Allows tract demand to be served at most once
                MMmodel.addConstrs((zij.sum(tract, '*') <= 1 for tract in tracts), name="Serve if pharmacy open")

                #Cannot stop at a pharmacy if that pharmacy is more than max_time hours from starting clinic
                #(To avoid stop-overs for for longer drives)
                reachable_pharm_consts = MMmodel.addConstrs(yp[pharm] == 0 for pharm in yp) #Guaranteed to be overwritten

                ########################
                ## Defines objective  ##
                ########################
                MMmodel.setObjective(zij.prod(demand_served) - eps*xij.sum('*','*'), GRB.MAXIMIZE) #Maximize number of unserved demand met minus epsilon edge penalty term

                #Universal parameters
                MMmodel.Params.lazyConstraints = 1
                MMmodel.Params.TimeLimit = 10*60 # 10 minute time limit (can revisit if needed)

                ############################################################
                ## Loops over starting location and solves model for each ##
                ############################################################
                for start_loc in clinics:
                    tic = time.perf_counter()

                    #Write initial results files with headers
                    Route_df = pd.DataFrame(columns = ["Origin","Destination","Demand Served","Max Demand","Drive Time","Setup Time","Service Time"])						   

                    #Remove prior start location
                    MMmodel.remove(start_loc_1_const)
                    MMmodel.remove(start_loc_2_const)

                    #Add new start location
                    start_loc_1_const = MMmodel.addConstr(xij.sum(start_loc,'*') == 1, name="Selected clinic start") 
                    start_loc_2_const = MMmodel.addConstr(xij.sum('*',start_loc) == 1, name="Selected clinic start") 

                    #Find which pharmacies are reachable from this clinic
                    reachable_pharms_data = cp_distances_data[cp_distances_data["Origin"] == start_loc]
                    reachable_pharms_data = reachable_pharms_data[reachable_pharms_data["Drive_Time"] <= max_dist]
                    reachable_pharms = reachable_pharms_data['Destination'].unique().tolist()
                    MMmodel.remove(reachable_pharm_consts)
                    reachable_pharm_consts = MMmodel.addConstrs(yp[pharm] == 0 for pharm in yp if pharm not in reachable_pharms)
                                        
                    #Optimize
                    MMmodel._xij = xij
                    MMmodel.update() #Necessary?
                    MMmodel.optimize(subtourelim)
                    toc = time.perf_counter()

                    #####################
                    ## Records results ##
                    #####################
                    
                    #Checks if model infeasible
                    if MMmodel.status == GRB.OPTIMAL:
                    
                        # Extract solution
                        solution_yp = MMmodel.getAttr('x', yp)
                        solution_xij = MMmodel.getAttr('x', xij)
                        solution_zij = MMmodel.getAttr('x', zij)

                        route = gp.tuplelist((origin, destination) for origin, destination in solution_xij.keys() if solution_xij[origin, destination] > 0.5)

                        for loc in route:
                            origin = loc[0]
                            destination = loc[1]
                            if destination in pharmacies:
                                demand_served_result = sum(solution_zij[tract,pharm]*demand_served[(tract,pharm)] for (tract,pharm) in zij if pharm == destination)
                                max_demand = sum(demand_served[(tract,pharm)] for (tract,pharm) in zij if pharm == destination)
                                Setup_time = setup_time
                                Service_time = demand_served_result*serve_time
                            else:
                                demand_served_result = max_demand = Setup_time = Service_time = nan


                            Route_df = Route_df.append(pd.DataFrame({'Origin':origin,
                                                                    'Destination':destination,
                                                                    'Demand Served':demand_served_result,
                                                                    'Max Demand':max_demand,
                                                                    'Drive Time':drive_times[(origin,destination)],
                                                                    'Setup Time':Setup_time,
                                                                    'Service Time':Service_time}, index=[0]))

                        # Saves route to csv
                        Route_df.to_csv(path_or_buf="/home/bonifontea/Mobile_Methadone/Results/Optimization_12_07_22_Nationwide/Routes/Routes/" + state + "Setup_" + str(setup_time) + "_Serve_" + str(serve_time) + "_Travel_" + str(max_travel) + "Route_starting_at_" + start_loc + ".csv",
                                                            sep=",",
                                                            index = False)
                                                        
                        # Create map with folium
                        tours = subtour(route) #Assumes only a single tour, no subtours
                        
                        #Centers map on starting clinic
                        Lat_center = clinic_data.at[start_loc,'Latitude']
                        Long_center = clinic_data.at[start_loc,'Longitude']

                        map = folium.Map(location=[Lat_center,Long_center], zoom_start = 8)

                        points = []
                        for loc in tours:
                            if loc in pharmacies:
                                demand_served_result = sum(solution_zij[tract,pharm]*demand_served[(tract,pharm)] for (tract,pharm) in zij if pharm == loc)
                                points.append([state_pharmacies.at[loc,'Latitude'],state_pharmacies.at[loc,'Longitude']])
                                folium.Marker([state_pharmacies.at[loc,'Latitude'],state_pharmacies.at[loc,'Longitude']],
                                    popup="Demand served: "+ str(round(demand_served_result))).add_to(map)
                            elif loc != 'Dummy':			
                                points.append([clinic_data.at[loc,'Latitude'],clinic_data.at[loc,'Longitude']])
                                folium.Marker([clinic_data.at[loc,'Latitude'],clinic_data.at[loc,'Longitude']],icon=folium.Icon(color="red")).add_to(map)
                        
                        #Add missing edge
                        points.append(points[0])
                            
                        folium.PolyLine(points).add_to(map)

                        map.save("/home/bonifontea/Mobile_Methadone/Results/Optimization_12_07_22_Nationwide/Routes/Maps/" + state + "Setup_" + str(setup_time) + "_Serve_" + str(serve_time) + "_Travel_" + str(max_travel) + "Route_starting_at_" + start_loc + ".html")

                        #Results
                        these_results_df = pd.DataFrame({'State':state,
                                                          'Starting Clinic':start_loc,
                                                          'Setup Param': setup_time, 
                                                          'Service Param': serve_time, 
                                                          'Max Travel Param': max_travel,
                                                          'Pharmacy Stops':[round(solution_yp.sum().getValue())],
                                                          'Demand Served':[solution_zij.prod(demand_served).getValue()],
                                                          'Drive Time': [solution_xij.prod(drive_times).getValue()],
                                                          'Service Time':[solution_zij.prod(demand_served).getValue()*serve_time],
                                                          'Setup Time':[solution_yp.sum().getValue()*setup_time],
                                                          'Computational Run Time':[toc-tic]})

                        these_results_df.to_csv(path_or_buf="/home/bonifontea/Mobile_Methadone/Results/Optimization_12_07_22_Nationwide/Aggregate_results.csv",
                                                    sep=",",
                                                    mode='a',
                                                    header=False,
                                                    index = False)
                                                    
                
                    else:
                        #Results
                        these_results_df = pd.DataFrame({'State':state,
                                                          'Starting Clinic':start_loc,
                                                          'Setup Param': setup_time, 
                                                          'Service Param': serve_time, 
                                                          'Max Travel Param': max_travel,
                                                          'Pharmacy Stops':'Infeasible',
                                                          'Demand Served':'Infeasible',
                                                          'Drive Time': 'Infeasible',
                                                          'Service Time':'Infeasible',
                                                          'Setup Time':'Infeasible',
                                                          'Computational Run Time':[toc-tic]})

                        these_results_df.to_csv(path_or_buf="/home/bonifontea/Mobile_Methadone/Results/Optimization_12_07_22_Nationwide/Aggregate_results.csv",
                                                    sep=",",
                                                    mode='a',
                                                    header=False,
                                                    index = False)