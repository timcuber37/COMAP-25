import pandas as pd

def readFiles():
    #retrieves the data from the files
    athleteD = pd.read_csv('2025_Problem_C_Data/2025_Problem_C_Data/summerOly_athletes.csv')
    hostD = pd.read_csv('2025_Problem_C_Data/2025_Problem_C_Data/summerOly_hosts.csv')
    medalD = pd.read_csv('2025_Problem_C_Data/2025_Problem_C_Data/summerOly_medal_counts.csv')
    programD = pd.read_csv('2025_Problem_C_Data/2025_Problem_C_Data/summerOly_programs.csv')

    return athleteD, hostD, medalD, programD



