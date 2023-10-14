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
