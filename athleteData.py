import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#retrieves the data from the files
athleteD = pd.read_csv('2025_Problem_C_Data/2025_Problem_C_Data/summerOly_athletes.csv')
"""hostD = pd.read_csv('2025_Problem_C_Data/2025_Problem_C_Data/summerOly_hosts.csv')
medalD = pd.read_csv('2025_Problem_C_Data/2025_Problem_C_Data/summerOly_medal_counts.csv')
programD = pd.read_csv('2025_Problem_C_Data/2025_Problem_C_Data/summerOly_programs.csv')
"""
athlete_y = athleteD['Medal'].replace(to_replace=['No medal', 'Bronze', 'Silver','Gold'], value= [0, 1, 2, 3])

athlete_x = athleteD.drop(axis = 1, labels = 'Medal')

print(athlete_x)
print(athlete_y)

x_train, x_test, y_train, y_test = train_test_split(athlete_x, athlete_y, train_size=.8, random_state=42)





