import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts, KFold as KF
from sklearn.ensemble import RandomForestClassifier as RFC, RandomForestRegressor as RFR
from sklearn.metrics import accuracy_score as ac, root_mean_squared_error as rmse, RocCurveDisplay as rcd, PredictionErrorDisplay as ped, r2_score as r2
import matplotlib.pyplot as plt

def getIntervals(actual, predictions):
    residuals = np.abs(actual - predictions)
    upperResidual = predictions + residuals
    lowerResidual = predictions - residuals
    return upperResidual, lowerResidual

def getResiduals(actual, predictions):
    residuals = np.abs(actual - predictions)
    return residuals

def countries(data, modelPlot = True, futureData = None, predict = ''):
    #Regression model for country data
    country_y = data[predict]
    country_x = data.drop(axis = 1, labels = ['Gold', 'Silver', 'Bronze', 'Total']).map(hash)  #split data

    x_train, x_test, y_train, y_test = tts(country_x, country_y, train_size =.8, random_state=2)

    model_rfr = RFR(n_estimators = 200, random_state=2)
    model_rfr.fit(x_train, y_train)

    y_pred = model_rfr.predict(x_test)
    #y_pred = pd.DataFrame(y_pred)

    a = rmse(y_test, y_pred)
    b = r2(y_test, y_pred)

    print(f"{predict}\nRoot means squared: {a} \nR2 Score: {b}")

    if modelPlot == True:
        y_ub, y_lb = getIntervals(y_test, y_pred)
        
        #sns.lmplot(x=y_pred, y=y_test, ci = 95)
        plt.scatter(y_pred, y_test, color= 'blue', label = 'Predictions')
        plt.scatter(y_lb, y_test, color= 'red', label = 'Lower Bound')
        plt.scatter(y_ub, y_test, color= 'green', label = 'Upper Bound')
        plt.title(f"Model Performance")
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        plt.show()
    
    if futureData is not None:
        futureData.drop(axis=1, labels= 'Year')
        future_pred = model_rfr.predict(futureData.map(hash))

        plt.figure(figsize=(20,10))
        plt.bar(x=futureData['NOC'], height=future_pred, color='gray')

        plt.title(f'Future {predict} Predictions')
        plt.xlabel('Competing Countries')
        plt.ylabel('Predicted Medal Counts')
        plt.xticks(rotation = 80)
        plt.show()

        predictions = {predict : future_pred}
        pred_df = pd.DataFrame(predictions)
        
        pred_df.to_csv(f'2025_Problem_C_Data/2025_Problem_C_Data/{predict}.csv', index = False)
