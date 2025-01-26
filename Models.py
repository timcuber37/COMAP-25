import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts, KFold as KF
from sklearn.ensemble import RandomForestClassifier as RFC, RandomForestRegressor as RFR
from sklearn.metrics import accuracy_score as ac, root_mean_squared_error as rmse, RocCurveDisplay as rcd, PredictionErrorDisplay as ped, r2_score as r2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn_quantile import RandomForestQuantileRegressor as RFQR

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

def intervals(model, X, percentile =95):
    lb = []
    ub = []
    for i in range(len(X)):
        predictions = []
        for pred in model.estimators_:
            predictions.append(pred.predict(X[i])[0])
        lb.append(np.percentile(predictions, (100 - percentile) / 2.))
        ub.append(np.percentile(predictions, 100  - (100-percentile) / 2.))
    return lb, ub

def countries(data, model = 1):
    #Regression model for country data
    country_y = data['Total']
    country_x = data.drop(axis = 1, labels = 'Total').map(hash)  #split data

    if model == 1: #prediction graphic

        x_train, x_test, y_train, y_test = tts(country_x, country_y, train_size =.8, random_state=2)

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

        x_train, x_test, y_train, y_test = tts(country_x, country_y, train_size = .8, random_state=2)
        
        model_rfr = RFR(n_estimators = 200, random_state=2)
        model_rfr.fit(x_train, y_train)
        
        y_pred = model_rfr.predict(x_test)
        y_pred = pd.DataFrame(y_pred)

        lb, ub = intervals(model_rfr, x_test)

        print(f"Lower Bound: {lb}\nUpper Bound{ub}")

    if model == 3:

        x_train, x_test, y_train, y_test = tts(country_x, country_y, train_size = .8, random_state=2)

        model_rfqr = RFQR()

