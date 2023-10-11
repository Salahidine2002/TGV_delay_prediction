"""
Main Python module launching the pipeline to assess the delay of the TGV.

Functions
---------
"""
from tools.tools_database import Read_data 
from tools.tools_preprocessing import *

#%%  Tests

Dataset = Read_data('./Data/TGV.csv')

print(check_for_same_departure_arrival_station(Dataset))
print(check_for_same_trip_in_same_month(Dataset))
   