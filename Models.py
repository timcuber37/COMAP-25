import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier as RFC, RandomForestRegressor as RFR
from sklearn.metrics import accuracy_score as ac, root_mean_squared_error as rmse, RocCurveDisplay as rcd, PredictionErrorDisplay as ped, r2_score as r2
import matplotlib.pyplot as plt
import seaborn as sns

#retrieves the data from the files
athleteD = pd.read_csv('2025_Problem_C_Data/2025_Problem_C_Data/summerOly_athletes.csv')

countryD = pd.read_csv('2025_Problem_C_Data/2025_Problem_C_Data/summerOly_medal_counts.csv')

def athletes(data, plot = True): 
    #Classification model for athlete Data
    athlete_y = data['Medal'].replace(['Gold', 'Silver', 'Bronze', 'No medal'], [1, 1, 1, 0]) #y is binary outcome (1 = medal, 0 = no medal)
    athlete_x = data.drop(axis = 1, labels = 'Medal').map(hash) #x is all other imputs

    x_train, x_test, y_train, y_test = tts(athlete_x, athlete_y, train_size=.2, random_state=42) #Split data

    model_rfc = RFC(n_estimators = 100, random_state=42)
    model_rfc.fit(x_train, y_train)

    y_pred = model_rfc.predict(x_test)
    y_pred = pd.DataFrame(y_pred)

    a = rmse(y_test, y_pred)
    b = ac(y_test, y_pred)

    print(f"Root means squared: {a} \n Accuracy Score: {b}")

    if plot is True: #prediction graphic *beware this sucks*
        ax = plt.gca()
        rfc_disp = rcd.from_estimator(model_rfc, x_test, y_test, ax=ax, alpha = .8)
        rfc_disp.plot(ax=ax, alpha = .8)
        plt.show()

def countries(data, plot = True):
    #Regression model for country data
    country_y = data['Total']
    country_x = data.drop(axis = 1, labels = 'Total').map(hash)

    x_train, x_test, y_train, y_test = tts(country_x, country_y, train_size=.2, random_state=42) #split data

    model_rfr = RFR(n_estimators = 100, random_state=42)
    model_rfr.fit(x_train, y_train)

    y_pred = model_rfr.predict(x_test)
    y_pred = pd.DataFrame(y_pred)

    a = rmse(y_test, y_pred)
    b = r2(y_test, y_pred)

    print(f"Root means squared: {a} \nR2 Score: {b}")

    if plot is True: #prediction graphic
        ax = plt.gca()
        rfc_disp = ped.from_estimator(model_rfr, x_test, y_test, kind = 'actual_vs_predicted')
        rfc_disp.plot()
        plt.show()


countries(countryD)

#athletes(athleteD)
