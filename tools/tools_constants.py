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

### Paths ###

PATH_DATASET = "Data/TGV.csv"

### Features ###

DELAY_FEATURE = "retard_moyen_arrivee"
LIST_CAUSE_FEATURES = [
    "prct_cause_externe",
    "prct_cause_infra",
    "prct_cause_gestion_trafic",
    "prct_cause_materiel_roulant",
    "prct_cause_prise_en_charge_voyageurs",
]

quant_features = ['duree_moyenne', 'nb_train_prevu', 'nb_annulation', 'nb_train_depart_retard',
                  'retard_moyen_depart', 'retard_moyen_tous_trains_depart', 'nb_train_retard_arrivee']

dropped_cols = ['commentaire_annulation',
                'commentaire_retards_depart', 'commentaires_retard_arrivee']

alph = 1.0
tolerance = 0.0001
iter_max = 1000
l1_ratio = 0.5
