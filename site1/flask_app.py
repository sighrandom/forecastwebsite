from flask import Flask, request, session, make_response, render_template, send_file

import pmdarima
import pandas
import processing
import statsmodels.api
import numpy

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/forecast', methods =["GET","POST"])

def home():
    # if method is post, then read the submitted file and calculate forecast results
    if request.method =="POST":
        # Read the file and initiate parameters for forecasting
        if request.form["action"] == "Use demo file":
            input_file = pandas.read_csv('/home/leojyz/site1/data.csv')
            date_col = 0
            value_col = 1
        if request.form["action"] == "Submit":
            input_file = pandas.read_csv(request.files.get('input'))
            date_col = int(request.form["date_column"]) - 1
            value_col = int(request.form["value_column"]) - 1

        try:
            forecast_timeframe = int(request.form["forecast_timeframe"])
        except:
            forecast_timeframe = 12

        # Retrieve only the Date and Value columns and set Date as Index and set seasonality
        data = input_file[[input_file.columns[date_col], input_file.columns[value_col]]]
        data.columns = ['Date','Value']
        data['Date'] = pandas.to_datetime(data['Date'])
        data = data.set_index('Date')
        data.index.freq = pandas.infer_freq(data.index)
        if data.index.freq == 'QS' or data.index.freq == 'Q':
            seasonality = 4
        elif data.index.freq == 'MS' or data.index.freq == 'M':
            seasonality = 12
        elif str(data.index.freq)[1] == 'W':
            seasonality = 52
        else:
            seasonality = 12

        # Splitting the training and test datasets
        data_train_df = data.iloc[:-forecast_timeframe]
        data_test_df = data.iloc[-forecast_timeframe:]

        # Initialising dataframes for storing model assessment outputs
        forecast_df = data_test_df.copy()
        fit_df = data_train_df.copy()
        residuals_df = data_train_df.copy()
        accuracy_df = pandas.DataFrame(index=['MAPE','RMSE'])

        # Initialising dataframe for storing final output
        final_df = pandas.DataFrame(index=pandas.date_range(start=max(data.index),
                                                            periods=forecast_timeframe+1,
                                                            freq=data.index.freq)).iloc[-forecast_timeframe:]



        # Compute naive forecasts
        fit_df['Naive'] = processing.seasonal_naive(data_train_df['Value'], seasonality, forecast_timeframe)[0]
        forecast_df['Naive'] = processing.seasonal_naive(data_train_df['Value'], seasonality, forecast_timeframe)[1].values
        residuals_df['Naive'] = data_train_df['Value'] - fit_df['Naive']
        accuracy_df['Naive'] = processing.accuracy(data_test_df['Value'],forecast_df['Naive'])
        naive_residual_checks_table = processing.residual_checks(residuals_df['Naive'].dropna(), seasonality)

        final_df['Naive'] = processing.seasonal_naive(data['Value'],seasonality,forecast_timeframe)[1].values


        # Compute ETS Log AdA forecast
        data_train_log = numpy.log(data_train_df['Value'])
        ETS = statsmodels.api.tsa.statespace.ExponentialSmoothing(data_train_log, trend=True, initialization_method='heuristic',
                                                          seasonal=seasonality, damped_trend=True).fit()
        fit_df['ETS'] = numpy.exp(ETS.fittedvalues)
        forecast_df['ETS'] = numpy.exp(ETS.forecast(forecast_timeframe))
        residuals_df['ETS'] = data_train_df['Value'] - fit_df['ETS']
        accuracy_df['ETS'] = processing.accuracy(data_test_df['Value'],forecast_df['ETS'])
        ETS_residual_checks_table = processing.residual_checks(residuals_df['ETS'].dropna(), seasonality)

        data_log = numpy.log(data['Value'])
        ETS_final = statsmodels.api.tsa.statespace.ExponentialSmoothing(data_log, trend=True, initialization_method='heuristic',
                                                          seasonal=seasonality, damped_trend=True).fit()
        final_df['ETS'] = numpy.exp(ETS_final.forecast(forecast_timeframe))

        # Compute SARIMA forecast
        SARIMA_AIC_test = pmdarima.auto_arima(data_train_df['Value'],
                                              seasonal=True,
                                              m=seasonality,
                                              d=1,
                                              information_criterion='aicc')
        SARIMA_model = statsmodels.tsa.statespace.sarimax.SARIMAX(endog=data_train_df['Value'],
                                                                  order = SARIMA_AIC_test.order,
                                                                  seasonal_order= SARIMA_AIC_test.seasonal_order,
                                                                  trend='c',
                                                                  enforce_invertibility=False)
        SARIMA_fit = SARIMA_model.fit()
        fit_df['SARIMA'] = SARIMA_fit.fittedvalues
        forecast_df['SARIMA'] = SARIMA_fit.predict(len(data_train_df),
                                                   len(data_train_df)+forecast_timeframe-1,
                                                   dynamic=False)
        residuals_df['SARIMA'] = data_train_df['Value'] - fit_df['SARIMA']
        accuracy_df['SARIMA'] = processing.accuracy(data_test_df['Value'],forecast_df['SARIMA'])
        SARIMA_residual_checks_table = processing.residual_checks(residuals_df['SARIMA'].dropna(), seasonality)

        SARIMA_model_final = statsmodels.tsa.statespace.sarimax.SARIMAX(endog=data['Value'],
                                                                  order = SARIMA_AIC_test.order,
                                                                  seasonal_order= SARIMA_AIC_test.seasonal_order,
                                                                  trend='c',
                                                                  enforce_invertibility=False)
        SARIMA_fit_final = SARIMA_model_final.fit()
        final_df['SARIMA'] = SARIMA_fit_final.predict(len(data),
                                                      len(data)+forecast_timeframe-1,
                                                      dynamic=False)

        # Calculate final ensemble predictions
        final_df['Ensemble'] = final_df.mean(axis=1)
        final_tab = final_df['Ensemble'].to_frame()


        # Generate forecast line plots
        naive_img = processing.plot_line(data_train_df.index.array,data_train_df['Value'],
                                   data_test_df.index.array,data_test_df['Value'],
                                   fit_df.index.array,fit_df['Naive'],
                                   forecast_df.index.array,forecast_df['Naive'])

        ETS_img = processing.plot_line(data_train_df.index.array,data_train_df['Value'],
                                   data_test_df.index.array,data_test_df['Value'],
                                   fit_df.index.array,fit_df['ETS'],
                                   forecast_df.index.array,forecast_df['ETS'])

        SARIMA_img = processing.plot_line(data_train_df.index.array,data_train_df['Value'],
                                   data_test_df.index.array,data_test_df['Value'],
                                   fit_df.index.array,fit_df['SARIMA'],
                                   forecast_df.index.array,forecast_df['SARIMA'])

        final_img = processing.plot_final(data.index,data['Value'],
                                          final_df['Naive'],final_df['ETS'],final_df['SARIMA'],final_df['Ensemble'],
                                          final_df.index)
        
        #Render webpage displaying data, final results and submodel tests
        return render_template('view.html',
                               tab=data.to_html(classes='onlyone'),
                               tab_final=final_tab.to_html(classes='onlyone'),
                               tab5=accuracy_df.to_html(classes='onlyone'),
                               image=naive_img.decode('utf8'),
                               image2=ETS_img.decode('utf8'),
                               image3=SARIMA_img.decode('utf8'),
                               image4=final_img.decode('utf8'),
                               tab2=naive_residual_checks_table.to_html(classes='onlyone',index=False),
                               tab3=ETS_residual_checks_table.to_html(classes='onlyone', index=False),
                               tab4=SARIMA_residual_checks_table.to_html(classes='onlyone', index=False)
                               )

    #Render page for submitting inputs
    return '''
        <html>
            <body>
                <h>Input how many periods you want to forecast out:</h>
                <form method="post" action="/forecast" enctype="multipart/form-data">
                    <p>Forecast periods: <input name="forecast_timeframe"/></p>  
                    <p>Submit data file to use:                    
                    <p><input type="submit" name="action" value="Use demo file"/></p>
                    <p>OR</p>                    
                    <p><input type="file" name="input"/></p>
                    <p>Date column number: <input name="date_column"/></p>
                    <p>Value column number: <input name="value_column"/></p>  
                    <p><input type="submit" name="action" value="Submit"/></p>
                    <br>
                    <p>Details of use:</p>
                    <p>Only accepts csv file
                    <br>Must have a column of dates and a column of values that you want to forecast
                    <br>Please only submit quarterly or monthly dataseries, anything more frequent will run out of memory
                    (I don't want to spend money on extra)</p>
                </form>
            </body>
        </html>
    '''

#Some other apps for learning purposes
@app.route("/sumfile", methods=["GET","POST"])
def sumfile_page():
    if request.method == "POST":
        submission = request.files["submission"]
        input_data = submission.stream.read().decode("utf-8")
        output_data = processing.process_data(input_data)
        response = make_response(output_data)
        response.headers["Content-Disposition"] = "attachment; filename=yourfile.csv"
        return response

    return '''
        <html>
            <body>
                <p>Select the file that you want to sum up:</p>
                <form method="post" action="/sumfile" enctype="multipart/form-data">
                    <p><input type="file" name="submission"/></p>
                    <p><input type="submit" value="Process the file"/></>
                </form>
            </body>
        </html>
    '''


'''Sum page - to add up multiple different user supplied inputs'''
@app.route("/sum", methods=["GET","POST"])
def list_page():
    errors = ""
    if "inputs" not in session:
        session["inputs"] = []

    if request.method =="POST":
        try:
            session["inputs"].append(float(request.form["user_input"]))
            session.modified = True
        except:
            errors += "<p>{!r} is not a number.</p>\n".format(request.form["user_input"])

    if len(session["inputs"]) == 0:
        number_list = ""
    else:
        number_list = "<p>Numbers you have entered are:</p>"
        for x in session["inputs"]:
            number_list += "<p>{x}</p>".format(x=x)



        if request.form["action"] == "Calculate sum":
            result = processing.calculate_mean(session["inputs"])
            session["inputs"].clear() # how does inputs clearing solve the reset issue?
            session.modified = True
            return '''
                <html>
                    <body>
                        <p>{result}</p>
                        <p><a href="/sum">Click here to start again</a>
                    </body>
                </html>
            '''.format(result=result)

    return '''
        <html>
            <body>
                {previous}
                {errors}
                <p>Enter your number:</p>
                <form method="post" action="/sum">
                    <p><input name="user_input" /></p>
                    <p><input type="submit" name="action" value="Add number" /></p>
                    <p><input type="submit" name="action" value="Calculate sum" /></p>
                </form>
            </body>
        </html>
    '''.format(errors=errors,previous=number_list)

'''Average page - for finding the average of two numbers'''
@app.route("/average", methods=["GET", "POST"])
def average_page():

    errors = ""
    if request.method =="POST":
        number1 = None
        number2 = None
        try:
            number1 = float(request.form["numberA"])
        except:
            errors += "<p>{!r} is not a number.</p>\n".format(request.form["numberA"])
        try:
            number2 = float(request.form["number2"])
        except:
            errors += "<p>{!r} is not a number.</p>\n".format(request.form["number2"])

        if number1 is not None and number2 is not None:
            result = processing.do_calculation(number1,number2)
            return '''
                <html>
                    <body>
                        <p>The result is {result}</p>
                        <p><a href="/list">Click here to calculate again</a>
                    </body>
                </html>
            '''.format(result=result)

    return '''
        <html>
            <body>
                {errors}
                <p>Enter your numbers:</p>
                <form method="post" action="/average">
                    <p><input name="numberA" /></p>
                    <p><input name="number2" /></p>
                    <p><input type="submit" value="Do calculation" /></p>
                </form>
            </body>
        </html>
    '''.format(errors=errors)
