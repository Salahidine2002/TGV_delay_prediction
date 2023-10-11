"""
Python module cleaning the data when necessary, after their validation.

Functions
---------
"""

#%% Functions definition

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

