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



def read_data(path) : 
    """
    Reads the data from a csv file's path.

    Parameters
    ----------
    path : str 
        Path to the dataset.

    Returns
    -------
    dataset : Pandas dataframe. 
        loaded dataset.
    """
    dataset = pd.read_csv(path, delimiter=';')
    dataset['date'] = pd.to_datetime(dataset['date'])
    return dataset


def load_coords(path='./Data/Coords.pickle') : 

    """
    Loads the dictionary where we stored the spherical coordinates of each station.

    Parameters
    ----------
    path : str 
        path to the pickle file where the dictionary is saved.

    Returns
    -------
    station_coordinates : python dict 
        In the format {key->station_name : value->coordinates, }.
    """
    with open(path, 'rb') as file:
        station_coordinates = pickle.load(file)

    return station_coordinates

def build_network(station_coordinates, dataset, s) : 

    """
    Construct the adjacency matrix of all stations in our dataset and the existing trajects.

    Parameters
    ----------
    station_cooridinates : python dict
        dictionary with the coordinates of each station.
    dataset : pandas.core.frame.DataFrame
        Dataset that contains all the possible trajects.
    s : list 
        Stations names list.

    Returns
    -------
    network : numpy ndrray 
        Adjacency matrix of all the stations network.
    """
    network = np.zeros((len(station_coordinates), len(station_coordinates))) 

    for i in range(len(network)) :
        for j in range(i) : 
            if len(dataset[dataset['gare_depart']==s[i]][dataset['gare_arrivee']==s[j]]) or  len(dataset[dataset['gare_depart']==s[j]][dataset['gare_arrivee']==s[i]]): 
                network[i, j], network[j, i] = 1, 1

    return network

def display_map(zoom=5): 

    """
    Construct the initial map centred around France.

    Parameters
    ----------
    zoom : int
        Initial map zoom.

    Returns
    -------
    france_map : 
        The map . 
    """

    france_map = folium.Map(location=[46.603354, 1.888334], zoom_start=zoom)

    return france_map 

def add_map_markers(map, markers_coords) : 

    """
    Adds a layer of markers (in the specified coordinates) on the given map in the parameters.

    Parameters
    ----------
    map : html èvjlv
        Initial empty map centred around France.
    markers_coords : python dict 
        Dictionary with the name of each marker (or station) and its coordinates

    """

    for station, coordinates in markers_coords.items():
        folium.Marker(location=coordinates, popup=station).add_to(map)


def add_map_routes(map, network, station_coordinates, s) : 

    """
    Adds a layer of lines for each traject in the network on the given map in the parameters.

    Parameters
    ----------
    map : html èvjlv
        Initial empty map centred around France.
    network : numpy ndarray 
        Adjacency matrix for our network 
    station_coordinates : numpy dict 
        Stations coordinates dictionary.
    s : list
        stations names list

    """

    for i in range(len(s)) : 
        for j in range(i) : 
            if network[i, j] : 
                coordinates = [station_coordinates[s[i]], station_coordinates[s[j]]]
                folium.PolyLine(locations=coordinates, color='black', weight=1).add_to(map)

    

def display_network(dataset) : 
    """
    displays the network of all the trajects specified in the dataset on a map 

    Parameters
    ----------
    dataset : pandas.core.frame.DataFrame
        Dataset that contains all the possible trajects.

    Returns
    -------
    france_map : 
        The map .
    """

    station_coordinates = load_coords()
    s = list(station_coordinates.keys())
    
    network = build_network(station_coordinates, dataset, s)

    france_map = display_map(5)
    
    add_map_markers(france_map, station_coordinates) 

    add_map_routes(france_map, network, station_coordinates, s)

    return france_map 


def display_map_delays(dataset, column='delay') : 

    """
    displays the means of one column with respect to each station on a map 

    Parameters
    ----------
    dataset : pandas.core.frame.DataFrame
        Dataset that contains all the possible trajects.
    column : str 
        The numeric feature to display
    
    Returns
    -------
    france_map : 
        The map .
    """

    station_coordinates = load_coords()
    s = list(station_coordinates.keys())

    france_map = display_map(5)

    delays = []

    for station in s : 

        frame = dataset[dataset['gare_arrivee']==station]

        if column=='delay' : 
            delays.append(np.mean(frame['retard_moyen_arrivee']/frame['duree_moyenne']))
        else : 
            delays.append(np.mean(frame[column]))

    scaler = MinMaxScaler(feature_range=(0, 1))
    delays = scaler.fit_transform(np.array(delays).reshape(-1, 1))

    radius = 70000*delays.flatten() 

    for i in range(len(s)) : 
        folium.Circle(
            location=station_coordinates[s[i]],
            radius=radius[i],
            color='blue',  # Circle border color
            weight=1,  
            fill=True,
            fill_color='blue',  # Circle fill color
            fill_opacity=0.1,   # Opacity of the fill color
        ).add_to(france_map)

    return france_map 


def box_plot_months(dataset, gare_dep, gare_arr, column) : 

    """
    The delay's (or any other numerical feature) box plot on one traject with respect to months 

    Parameters
    ----------
    dataset : pandas.core.frame.DataFrame
        Dataset that contains all the possible trajects.
    gare_dep : str 
        Name of departure train station 
    gare_arr : str 
        Name of arriving train station 
    column : str 
        Numerical feature to plot
    
    """

    data = []
    months = np.unique(dataset['date'].dt.month)

    for month in months: 

        frame = dataset[dataset['gare_depart']==gare_dep]
        frame = frame[frame['gare_arrivee']==gare_arr]
        frame = frame[frame['date'].dt.month==month]

        data.append(np.array(frame[column])) 
        
    plt.boxplot(data, labels=months)
    plt.xlabel('Months')
    plt.ylabel(column)
    plt.title(f'Traject {gare_dep} / {gare_arr}')
    plt.show()

def histograms(dataset, columns) :

    """
    Displays the histograms of one or multiple numerical features

    Parameters
    ----------
    dataset : pandas.core.frame.DataFrame
        Dataset that contains all the possible trajects.
    column : list 
    List of the numerical features to plot their histograms 
    
    """

    dataset[columns].hist(figsize=(20, 20), bins=100)
    plt.show()

def remove_outliers(dataset, threshold) :

    """
    Removes the oultiers rows with respect to the difference between arival and departure delays

    Parameters
    ----------
    dataset : pandas.core.frame.DataFrame
        Dataset that contains all the possible trajects.
    threshold : float 
        Z score threshold to eliminate the outliers.
    """

    Col = dataset['retard_moyen_arrivee']-dataset['retard_moyen_depart']
    mean = np.mean(np.array(Col))
    std = np.std(np.array(Col))

    Z_score = abs(np.array((Col-mean)/std))

    dataset = dataset[Z_score<threshold]
    return dataset


        