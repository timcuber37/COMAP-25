from Model import *

countryD = pd.read_csv('2025_Problem_C_Data/2025_Problem_C_Data/summerOly_medal_counts.csv')
futureData = pd.read_csv('2025_Problem_C_Data/2025_Problem_C_Data/2028Rankings.csv')

futureSorted = futureData.sort_values(by= 'Rank')

medals = ['Gold', 'Silver', 'Bronze', 'Total']

for medal in medals:
    countries(countryD, futureData=futureSorted, predict=medal)