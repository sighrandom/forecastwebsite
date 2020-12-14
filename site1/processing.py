'''import pre-built packages'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
from io import BytesIO
import numpy
import pandas
import base64
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


# Define mean percentage
def MAPE(y_obs, y_pred):

    y_obs = numpy.array(y_obs)
    y_pred = numpy.array(y_pred)

    return numpy.mean(numpy.abs((y_obs-y_pred)/y_obs))*100

# Define root mean square error
def RMSE(y_obs,y_pred):

    y_obs = numpy.array(y_obs)
    y_pred = numpy.array(y_pred)

    return numpy.sqrt(numpy.mean((y_obs-y_pred)**2))

# Calculate the MAPE and RMSE for provided predictions and actuals
def accuracy(y_obs,y_pred):

    rmserror = numpy.round(RMSE(y_obs,y_pred),1)
    maperror = numpy.round(MAPE(y_obs,y_pred),1)

    return maperror, rmserror

#Calculate naive seasonal forecast by taking the last seasonal cycle and replicating it over the forecast_timeframe
def seasonal_naive(training_data,seasonal_period,forecast_timeframe):

    latest_season = training_data.iloc[-seasonal_period:]
    repetitions = numpy.int(numpy.ceil(forecast_timeframe/seasonal_period))
    forecast_array = numpy.tile(latest_season,repetitions)
    forecast = pandas.Series(forecast_array[:forecast_timeframe])
    fitted = training_data.shift(seasonal_period)

    return fitted, forecast

#Calculate series of tests for residuals and returns summary table:
#        Correlated if Ljung Box Test p-value <0.05
#        Not normally distributed if Jarque Bera Normality Test <0.05
#        Stationary if AD Fuller p-value <0.05
def residual_checks(residuals,seasonal_period):

    lags = min(len(residuals)/5,2*seasonal_period)
    residual_mean = numpy.mean(residuals)
    ljung_p_value = round(numpy.mean(acorr_ljungbox(x=residuals,lags=lags)[1]),3)
    normal_p_value = round(jarque_bera(residuals)[1],3)
    adfuller_p_value = round(adfuller(residuals)[1],3)

    if ljung_p_value <0.05:
        ljung_test = "Residuals are correlated"
    else:
        ljung_test = "Residuals are uncorrelated"

    if normal_p_value < 0.05:
        normal_test = "Residuals are NOT normally distributed"
    else:
        normal_test = "Residuals are normally distributed"

    if adfuller_p_value < 0.05:
        adfuller_test = "Timeseries is stationary"
    else:
        adfuller_test = "Timeseries is NOT stationary"

    data = {'Type': ['Ljung Box test p-value','Jarque Bera Normality Test','AD Fuller stationary test'],
            'Value': [ljung_p_value,normal_p_value,adfuller_p_value],
            'Conclusion': [ljung_test,normal_test,adfuller_test]
            }

    residual_table = pandas.DataFrame(data=data,columns = ['Type','Value','Conclusion'])

    return residual_table

# Generate scatter plot of two equal length series
def plot_scatter(x,y):

    plot.clf()
    plot.scatter(x, y, alpha=0.5)

    img = BytesIO()
    plot.savefig(img, format='png')
    img.seek(0)
    img_png = base64.b64encode(img.getvalue())

    return img_png

# Generate standard train-fit and forecast-test charts
def plot_line(train_x,train_y,test_x,test_y,fit_x,fit_y,forecast_x,forecast_y):

    plot.clf()
    plot.plot(train_x, train_y, label = 'Training set', color='blue')
    plot.plot(test_x,test_y, label = 'Test set',linestyle='--',marker = '.', color='blue')
    plot.plot(fit_x,fit_y,color='red', label = 'Fit')
    plot.plot(forecast_x,forecast_y,color='red',linestyle='--',marker = '.',label = 'Forecast')
    plot.legend()

    img = BytesIO()
    plot.savefig(img, format='png')
    img.seek(0)
    img_png = base64.b64encode(img.getvalue())

    return img_png

# Generate final chart with each model predictors
def plot_final(data_x,data_y,a_y,b_y,c_y,d_y,x):

    plot.clf()
    plot.plot(data_x, data_y, label = 'Observed', color='blue')
    plot.plot(x,a_y, label = 'Naive',linestyle='--',marker = '.',color='Orange')
    plot.plot(x,b_y, label = 'ETS',linestyle='--',marker = '.',color='Green')
    plot.plot(x,c_y, label = 'SARIMA',linestyle='--',marker = '.',color='Red')
    plot.plot(x,d_y, label = 'Final Ensemble',linestyle='--',marker = '.',color='blue')
    plot.legend()

    img = BytesIO()
    plot.savefig(img, format='png')
    img.seek(0)
    img_png = base64.b64encode(img.getvalue())

    return img_png