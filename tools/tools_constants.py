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

TEST_MODE = True

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

QUANT_FEATURES = ['duree_moyenne', 'nb_train_prevu', 'nb_annulation', 'nb_train_depart_retard',
                  'retard_moyen_depart', 'retard_moyen_tous_trains_depart', 'nb_train_retard_arrivee']

DROPPED_COLS = ['commentaire_annulation',
                'commentaire_retards_depart', 'commentaires_retard_arrivee']

### Others ###

RANDOM_STATE = 42

ALPH = 1.0
TOLERANCE = 0.0001
ITER_MAX = 1000
L1_RATIO = 0.5
