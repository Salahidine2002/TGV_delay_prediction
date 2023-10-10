"""
Python module cleaning the data when necessary, after their validation.

Functions
---------
"""
from tools_database import *
from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler, StandardScaler
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
import category_encoders as ce


class TransformerDrop(TransformerMixin, BaseEstimator):
    def __init__(self, to_drop):
        self.to_drop = to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Supprimer les colonnes
        X = X.drop(self.to_drop, axis=1)
        return X


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
