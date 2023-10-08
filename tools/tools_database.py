"""
Python module reading and visualizing the data.

Functions
---------
"""
import pandas as pd
import numpy as np 
import pickle
import folium
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



def Read_data(Path) : 
    Dataset = pd.read_csv(Path, delimiter=';')
    Dataset['date'] = pd.to_datetime(Dataset['date'])
    return Dataset


def Load_coords(Path='./Data/Coords.pickle') : 

    with open(Path, 'rb') as file:
        station_coordinates = pickle.load(file)

    return station_coordinates

def Build_network(station_coordinates, Dataset, S) : 

    # Building a network 

    Net = np.zeros((len(station_coordinates), len(station_coordinates))) 

    for i in range(len(Net)) :
        for j in range(i) : 
            if len(Dataset[Dataset['gare_depart']==S[i]][Dataset['gare_arrivee']==S[j]]) or  len(Dataset[Dataset['gare_depart']==S[j]][Dataset['gare_arrivee']==S[i]]): 
                Net[i, j], Net[j, i] = 1, 1

    return Net

def Display_map(zoom=5): 
    
    # Create a map centered around France
    france_map = folium.Map(location=[46.603354, 1.888334], zoom_start=zoom)

    return france_map 

def Add_map_markers(map, Markers_coords) : 

    # Create markers for each station and add them to the map
    for station, coordinates in Markers_coords.items():
        folium.Marker(location=coordinates, popup=station).add_to(map)


def Add_map_routes(map, Net, station_coordinates, S) : 

    for i in range(len(S)) : 
        for j in range(i) : 
            if Net[i, j] : 
                coordinates = [station_coordinates[S[i]], station_coordinates[S[j]]]
                folium.PolyLine(locations=coordinates, color='black', weight=1).add_to(map)

    

def Display_network(Dataset) : 

    station_coordinates = Load_coords()
    S = list(station_coordinates.keys())
    
    Net = Build_network(station_coordinates, Dataset, S)

    france_map = Display_map(5)
    
    Add_map_markers(france_map, station_coordinates) 

    Add_map_routes(france_map, Net, station_coordinates, S)

    return france_map 


def Display_map_delays(Dataset, column='delay') : 

    station_coordinates = Load_coords()
    S = list(station_coordinates.keys())

    france_map = Display_map(5)

    Delays = []

    for station in S : 

        Frame = Dataset[Dataset['gare_arrivee']==station]

        if column=='delay' : 
            Delays.append(np.mean(Frame['retard_moyen_arrivee']/Frame['duree_moyenne']))
        else : 
            Delays.append(np.mean(Frame[column]))

    scaler = MinMaxScaler(feature_range=(0, 1))
    Delays = scaler.fit_transform(np.array(Delays).reshape(-1, 1))

    Radius = 70000*Delays.flatten() 

    for i in range(len(S)) : 
        folium.Circle(
            location=station_coordinates[S[i]],
            radius=Radius[i],
            color='blue',  # Circle border color
            weight=1,  
            fill=True,
            fill_color='blue',  # Circle fill color
            fill_opacity=0.1,   # Opacity of the fill color
        ).add_to(france_map)

    return france_map 


def Box_plot_months(Dataset, gare_dep, gare_arr, column) : 

    Data = []
    Months = np.unique(Dataset['date'].dt.month)

    for month in Months: 

        Frame = Dataset[Dataset['gare_depart']==gare_dep]
        Frame = Frame[Frame['gare_arrivee']==gare_arr]
        Frame = Frame[Frame['date'].dt.month==month]

        Data.append(np.array(Frame[column])) 
        
    plt.boxplot(Data, labels=Months)
    plt.xlabel('Months')
    plt.ylabel(column)
    plt.title(f'Traject {gare_dep} / {gare_arr}')
    plt.show()