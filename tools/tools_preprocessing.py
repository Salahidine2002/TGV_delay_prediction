"""
Python module cleaning the data when necessary, after their validation.

Functions
---------

check_for_same_departure_arrival_station(Dataset)
    Function to check if a trip as the same departure and arrival station 

check_for_same_trip_in_same_month(Dataset)
    Function to check if a trip exists twice for the same month (it should not)
"""

###############
### Imports ###
###############

### Python imports ###

from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
import category_encoders as ce

### Module imports ###

from tools.tools_database import *
from tools.tools_constants import (
    QUANT_FEATURES,
    DROPPED_COLS
)

###############
### Classes ###
###############


class TransformerDrop(TransformerMixin, BaseEstimator):
    def __init__(self, to_drop):
        self.to_drop = to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Supprimer les colonnes
        X = last_month_column(X)

        X = X.drop(self.to_drop, axis=1)
        print(X)
        return X

# classe pour definir le transformer qui transforme les noms des gares
# en coordonnées x et y donc 2 features pour les gares d'entrée et 2 autres pour les gares de sortie
# L'utilisation de la class est pour pouvoir l'inclure par la suite dans la pipeline


class Transformercolonne(TransformerMixin, BaseEstimator):
    def __init__(self, to_transform):
        self.to_transform = to_transform

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # # Add the colummn of the predictions from last month
        # X = last_month_column(X)
        # Encode the coordinates of the train stations
        X = coords_encoding(X, self.to_transform)
        return X

#################
### Functions ###
#################

### Functions for encoding and normalisation ###


def drop(cols_to_drop):
    """
    drop the useless columns of a dataset

    Parameters
    ----------
    cols_to_drop : list of string
        The strings are the name of the columns

    Returns
    -------
    transformer_drop : TranformerDrop class
        Transformer which drop the columns of a dataset
    """
    transformer_drop = TransformerDrop(cols_to_drop)
    return transformer_drop


def dropped():
    """ call of the function drop for the columns 'dropped_cols' 
    dropped all the columns with commentary
    """
    return drop(DROPPED_COLS)


def pipeline_binary(scaling):
    """
    creation of the pre processing pipeline with, among others binary encoding for the stations 
    (including also dropping the useless columns, encoding the service and normalizing the features)
    Parameters
    ----------
    scaling : Transformer (class, sklearn.preprocessing)
        Transform features by scaling them

    Returns
    -------
    pipe : sklearn.pipeline.Make_pipeline
        Pipeline of the preprocessing with binary encoding for the stations
    """
    column_trans = ColumnTransformer(
        [('num', scaling, QUANT_FEATURES),
         ('cat_binary', ce.BinaryEncoder(),
          ['gare_depart', 'gare_arrivee']),
         ('cat_oh', OneHotEncoder(), ['service'])])
    pipe = make_pipeline(dropped(), column_trans)
    return pipe


# fonction qui réalise la transformation du dataset et des colonnes des gare en coordonnées (x et y)
def coords_encoding(Dataset, colonnes):
    """
    function which transform the name of columns (in this case it is used for stations)
    into their geographical coordinates

    Parameters
    ----------
    Dataset : pandas.core.frame.DataFrame
        Dataset to encode

    Returns
    -------
    dataset_to_encode : pandas.core.frame.DataFrame
        Dataset with the chosen columns encoded as their geographical coordiantes
    """
    load = load_coords(path='./Data/Coords.pickle')
    dataset_to_encod = Dataset

    gare_depart_coord_x = []
    gare_depart_coord_y = []
    gare_arrivee_coord_x = []
    gare_arrivee_coord_y = []

    for j in range(len(dataset_to_encod[colonnes[0]])):

        gare_depart_coord_x.append(
            load[dataset_to_encod.iloc[j][colonnes[0]]][0])
        gare_depart_coord_y.append(
            load[dataset_to_encod.iloc[j][colonnes[0]]][1])
        gare_arrivee_coord_x.append(
            load[dataset_to_encod.iloc[j][colonnes[1]]][0])
        gare_arrivee_coord_y.append(
            load[dataset_to_encod.iloc[j][colonnes[1]]][1])

    dataset_to_encod['gare_depart_coord_x'] = gare_depart_coord_x
    dataset_to_encod['gare_depart_coord_y'] = gare_depart_coord_y
    dataset_to_encod['gare_arrivee_coord_x'] = gare_arrivee_coord_x
    dataset_to_encod['gare_arrivee_coord_y'] = gare_arrivee_coord_y
    # TODO change place
    dataset_to_encod['date'] = dataset_to_encod["date"].dt.month

    del dataset_to_encod[colonnes[1]]
    del dataset_to_encod[colonnes[0]]

    return dataset_to_encod

# création de la pipeline qui encode en coordonné en fonction de la methode de normalisaton


def pipeline_coords(scaling):
    """
    creation of the pre processing pipeline with, among others coordinate encoding for the stations 
    (including also dropping the useless columns, encoding the service and normalizing the features)

    Parameters
    ----------
    scaling : Transformer (class, sklearn.preprocessing)
        Transform features by scaling them

    Returns
    -------
    pipe : sklearn.pipeline.Make_pipeline
        Pipeline of the preprocessing with geographical encoding for stations
    """
    column_trans = ColumnTransformer(
        [('num', scaling, QUANT_FEATURES), ('cat_oh', OneHotEncoder(), ['service'])])
    pipe = make_pipeline(dropped(), Transformercolonne(
        ['gare_depart', 'gare_arrivee']), column_trans)
    return pipe

# Function of the pipeline with binary encoding


def pipeline_minmax():
    """ 
    Creation of the pipeline with binary encoding for stations and MinMaxscaler scaling
    """
    return pipeline_binary(MinMaxScaler())


def pipeline_stand():
    """ 
    Creation of the pipeline with binary encoding for stations and Standardscaler scaling
    """
    return pipeline_binary(StandardScaler())


def pipeline_robust():
    """ 
    Creation of the pipeline with binary encoding for stations and Robustscaler scaling
    """
    return pipeline_binary(RobustScaler())

# Function of the pipeline with coordinate encoding


def pipeline_coords_robust():
    """ 
    Creation of the pipeline with coordinate encoding for stations and Robustscaler scaling
    """
    return pipeline_coords(RobustScaler())


def pipeline_coords_minmax():
    """ 
    Creation of the pipeline with coordinate encoding for stations and MinMaxscaler scaling
    """
    return pipeline_coords(MinMaxScaler())


def pipeline_coords_stand():
    """ 
    Creation of the pipeline with coordinate encoding for stations and Standardscaler scaling
    """
    return pipeline_coords(StandardScaler())

### Functions for data checking ###


def check_for_same_departure_arrival_station(Dataset):
    """
    Function to check if a trip as the same departure and arrival station 

    Parameters
    ----------
    dataset : pandas.core.frame.DataFrame
        Dataset where to analyse the correlation.

    Returns
    -------
    same_station :list
        list of ligne number of trips with same departure and arrival station
    """

    same_station = []
    for ligne in range(0, len(Dataset)-1):
        if (Dataset["gare_depart"][ligne] == Dataset["gare_arrivee"][ligne]):
            same_station.append(ligne)
    return (same_station)


def check_for_same_trip_in_same_month(Dataset):
    """
    Function to check if a trip exists twice for the same month (it should not)

    Parameters
    ----------
    dataset : pandas.core.frame.DataFrame
        Dataset where to analyse the correlation.

    Returns
    -------
    same_trip :list
        list of ligne number of same trips within one month
    """
    ma_list = []
    same_trip = []
    for ligne in range(len(Dataset)):
        ma_list.append([ligne, Dataset["date"].dt.to_period(
            'M')[ligne], Dataset["gare_depart"][ligne], Dataset["gare_arrivee"][ligne]])

    for i in range(0, len(ma_list)):
        for j in range(i+1, len(ma_list)):
            if ((ma_list[i][2] == ma_list[j][2]) and (ma_list[i][3] == ma_list[j][3]) and (ma_list[i][1] == ma_list[j][1])):
                same_trip.append(ma_list[i], ma_list[j])
    return (same_trip)
