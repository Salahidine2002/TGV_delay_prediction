"""
Python module containing the main constants of the code.

Constants
---------
PATH_DATASET : str
    Path to the dataset of the TGV delays.

DELAY_FEATURE : str
    Name of the feature representing the mean delay of the TGV at arrival.

LIST_CAUSE_FEATURES : list[str]
    List containing the name of the features representing the different causes of delay.
"""

#################
### Constants ###
#################

TEST_MODE = False

### Paths ###

PATH_DATASET = "Data/TGV.csv"
PATH_FIGURES = "figures/"

### Features ###

DELAY_FEATURE = "retard_moyen_arrivee"
LIST_CAUSE_FEATURES = [
    "prct_cause_externe",
    "prct_cause_infra",
    "prct_cause_gestion_trafic",
    "prct_cause_materiel_roulant",
    "prct_cause_prise_en_charge_voyageurs",
]
LIST_FEATURES_TRAINING = [
    "duree_moyenne",
    "nb_train_prevu",
    "gare_depart",
    "gare_arrivee",
    "service",
    "date",
    "retard_moyen_arrivee"
]

QUANT_FEATURES = ['duree_moyenne', 'nb_train_prevu']

DROPPED_COLS = [
    "retard_moyen_arrivee"
]

### Others ###

RANDOM_STATE = 42

ALPH = 1.0
TOLERANCE = 0.0001
ITER_MAX = 1000
L1_RATIO = 0.5
