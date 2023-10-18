"""
Python module cleaning the data when necessary, after their validation.

Functions
---------
"""

###############
### Imports ###
###############

from tools.tools_database import *
from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler, StandardScaler
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
import category_encoders as ce

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
        X = X.drop(self.to_drop, axis=1)
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
        # Supprimer les colonnes
        X = coords_encoding(X, self.to_transform)
        return X

#################
### Functions ###
#################

### Functions for encoding and normalisation ###


def drop(cols_to_drop):
    transformer_drop = TransformerDrop(cols_to_drop)
    return transformer_drop


def dropped():
    return drop(['commentaire_annulation', 'commentaire_retards_depart', 'commentaires_retard_arrivee'])


def pipeline_binary(scaling):
    quant_features = ['duree_moyenne', 'nb_train_prevu', 'nb_annulation', 'nb_train_depart_retard',
                      'retard_moyen_depart', 'retard_moyen_tous_trains_depart', 'nb_train_retard_arrivee']
    column_trans = ColumnTransformer(
        [('num', scaling, quant_features), ('cat_binary', ce.BinaryEncoder(), ['gare_depart', 'gare_arrivee']), ('cat_oh', OneHotEncoder(), ['service'])])
    pipe = make_pipeline(dropped(), column_trans)
    return pipe


# fonction qui réalise la transformation du dataset et des colonnes des gare en coordonnés (x et y)
def coords_encoding(Dataset, colonnes):
    L = load_coords(path='./Data/Coords.pickle')
    gare_depart_coord_x = []
    gare_depart_coord_y = []
    gare_arrivee_coord_x = []
    gare_arrivee_coord_y = []

    for j in range(len(Dataset[colonnes[0]])):

        gare_depart_coord_x.append(L[Dataset[colonnes[0]][j]][0])
        gare_depart_coord_y.append(L[Dataset[colonnes[0]][j]][1])
        gare_arrivee_coord_x.append(L[Dataset[colonnes[1]][j]][0])
        gare_arrivee_coord_y.append(L[Dataset[colonnes[1]][j]][1])

    Dataset['gare_depart_coord_x'] = gare_depart_coord_x
    Dataset['gare_depart_coord_y'] = gare_depart_coord_y
    Dataset['gare_arrivee_coord_x'] = gare_arrivee_coord_x
    Dataset['gare_arrivee_coord_y'] = gare_arrivee_coord_y

    del Dataset[colonnes[1]]
    del Dataset[colonnes[0]]

    return Dataset

# création de la pipeline qui encode en coordonné en fonction de la methode de normalisaton


def pipeline_coords(scaling):
    quant_features = ['gare_depart', 'gare_arrivee', 'duree_moyenne', 'nb_train_prevu', 'nb_annulation', 'nb_train_depart_retard',
                      'retard_moyen_depart', 'retard_moyen_tous_trains_depart', 'nb_train_retard_arrivee']
    column_trans = ColumnTransformer(
        [('num', scaling, quant_features), ('cat_oh', OneHotEncoder(), ['service'])])
    pipe = make_pipeline(dropped(), Transformercolonne(
        ['gare_depart', 'gare_arrivee']), column_trans)
    return pipe

# Function of the pipeline with binary encoding


def pipeline_minmax():
    return pipeline_binary(MinMaxScaler())


def pipeline_stand():
    return pipeline_binary(StandardScaler())


def pipeline_robust():
    return pipeline_binary(RobustScaler())

# Function of the pipeline with coordinate encoding


def pipeline_coords_robust():
    return pipeline_coords(RobustScaler())


def pipeline_coords_minmax():
    return pipeline_coords(MinMaxScaler())


def pipeline_coords_stand():
    return pipeline_coords(StandardScaler())

### Functions for data checking ###


def check_for_same_departure_arrival_station(Dataset):
    """
    Function to check if a trip as the same departure and arrival station 
    ----------
    """
    same_station = []
    for ligne in range(0, len(Dataset)-1):
        if (Dataset["gare_depart"][ligne] == Dataset["gare_arrivee"][ligne]):
            same_station.append(ligne)
    return (same_station)


def check_for_same_trip_in_same_month(Dataset):
    """
    Function to check if a trip exists twice for the same month (it should not)
    ----------
    """
    ma_list = []
    same_trip = []
    for ligne in range(len(Dataset)):
        ma_list.append([ligne, Dataset["date"].dt.to_period(
            'M')[ligne], Dataset["gare_depart"][ligne], Dataset["gare_arrivee"][ligne]])
    # print(len(ma_list))

    for i in range(0, len(ma_list)):
        for j in range(i+1, len(ma_list)):
            if ((ma_list[i][2] == ma_list[j][2]) and (ma_list[i][3] == ma_list[j][3]) and (ma_list[i][1] == ma_list[j][1])):
                # if ( ma_list[i][1] == ma_list[j][1]):
                same_trip.append(ma_list[i], ma_list[j])
    return (same_trip)
