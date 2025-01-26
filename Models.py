import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts, KFold as KF
from sklearn.ensemble import RandomForestClassifier as RFC, RandomForestRegressor as RFR
from sklearn.metrics import accuracy_score as ac, root_mean_squared_error as rmse, RocCurveDisplay as rcd, PredictionErrorDisplay as ped, r2_score as r2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn_quantile import RandomForestQuantileRegressor as RFQR

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

def f(x):
    return x * np.sin(x)

def countries(data, model = 1):
    #Regression model for country data
    country_y = data['Total']
    country_x = data.drop(axis = 1, labels = 'Total').map(hash)  #split data

    if model == 1: #prediction graphic

        x_train, x_test, y_train, y_test = tts(country_x, country_y, train_size =.2, random_state=2)

        model_rfr = RFR(n_estimators = 200, random_state=2)
        model_rfr.fit(x_train, y_train)

        y_pred = model_rfr.predict(x_test)
        y_pred = pd.DataFrame(y_pred)

        a = rmse(y_test, y_pred)
        b = r2(y_test, y_pred)

        print(f"Root means squared: {a} \nR2 Score: {b}")

        ped.from_estimator(model_rfr, x_test, y_test, kind = 'actual_vs_predicted')
        plt.show()

    if model == 2:

        x_train, x_test, y_train, y_test = tts(country_x, country_y, random_state=2)
        
        model_rfr = RFR(n_estimators = 200, random_state=2)
        model_rfr.fit(x_train, y_train)
        
        common_params = dict(max_depth=3, min_samples_leaf=4, min_samples_split=4)
        model_rfqr = RFQR(**common_params, q=[0.05, 0.5, 0.95])
        model_rfqr.fit(x_train, y_train)
        
        y_pred = model_rfr.predict(x_test)
        y_pred = pd.DataFrame(y_pred)
        
        rfqr_pred = model_rfqr.predict(x_train)
        
        y_lower = rfqr_pred[0]
        y_med = rfqr_pred[1]
        y_upper = rfqr_pred[2]

        x_test.to_numpy()

        fig = plt.figure(figsize=(10, 10))
        plt.plot(x_train, f(x_train), 'g:', linewidth=3, label=r'$f(x) = x\,\sin(x)$')
        plt.plot(x_test, y_test, 'b.', markersize=10, label='Test observations')
        plt.plot(x_train, y_med, 'r-', label='Predicted median', color="orange")
        plt.plot(x_train, y_pred, 'r-', label='Predicted mean')
        plt.plot(x_train, y_upper, 'k-')
        plt.plot(x_train, y_lower, 'k-')
        plt.fill_between(x_test.ravel(), y_lower, y_upper, alpha=0.4, label='Predicted 90% interval')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.ylim(-10, 25)
        plt.legend(loc='upper left')
        plt.show()

