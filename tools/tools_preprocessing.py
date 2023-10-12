"""
Python module cleaning the data when necessary, after their validation.

Functions
---------
"""

###############
### Imports ###
###############

from tools_database import *
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

#################
### Functions ###
#################

### Functions for encoding and normalisation ###

def drop(cols_to_drop):
    transformer_drop = TransformerDrop(cols_to_drop)
    return transformer_drop


def dropped():
    return drop(['commentaire_annulation', 'commentaire_retards_depart', 'commentaires_retard_arrivee'])


def pipeline(scaling):
    quant_features = ['duree_moyenne', 'nb_train_prevu', 'nb_annulation', 'nb_train_depart_retard',
                      'retard_moyen_depart', 'retard_moyen_tous_trains_depart', 'nb_train_retard_arrivee']
    column_trans = ColumnTransformer(
        [('num', scaling, quant_features), ('cat_binary', ce.BinaryEncoder(), ['gare_depart', 'gare_arrivee']), ('cat_oh', OneHotEncoder(), ['service'])])
    pipe = make_pipeline(dropped(), column_trans)
    return pipe


def pipeline_minmax():
    return pipeline(MinMaxScaler())


def pipeline_stand():
    return pipeline(StandardScaler())


def pipeline_robust():
    return pipeline(RobustScaler())

### Functions for data checking ###

def check_for_same_departure_arrival_station(Dataset):
    """
    Function to check if a trip as the same departure and arrival station 
    ----------
    """
    same_station = []
    for ligne in range (0,len(Dataset)-1) :
        if ( Dataset["gare_depart"][ligne] ==  Dataset["gare_arrivee"][ligne]) : 
            same_station.append(ligne)
    return(same_station)

def check_for_same_trip_in_same_month(Dataset):
    """
    Function to check if a trip exists twice for the same month (it should not)
    ----------
    """
    ma_list = []
    same_trip = []
    for ligne in range(len(Dataset)):
        ma_list.append([ligne, Dataset["date"].dt.to_period('M')[ligne], Dataset["gare_depart"][ligne], Dataset["gare_arrivee"][ligne]])
    #print(len(ma_list))

    for i in range(0, len(ma_list)):
        for j in range(i+1, len(ma_list)):
            if ( (ma_list[i][2] == ma_list[j][2]) and (ma_list[i][3] == ma_list[j][3]) and ( ma_list[i][1] == ma_list[j][1])) : 
                #if ( ma_list[i][1] == ma_list[j][1]):
                same_trip.append(ma_list[i], ma_list[j])
    return(same_trip)
