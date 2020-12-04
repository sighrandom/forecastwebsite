
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, request, session, make_response, render_template
import processing
import pandas

app = Flask(__name__)
app.config["DEBUG"] = True
app.config["SECRET_KEY"] = 'warfhjwoiuj348'


@app.route('/forecast', methods =["GET","POST"])
def home():

    if request.method =="POST":
        data = pandas.read_csv(request.files.get('input'))
        seasonality = 4
        forecast_horizon = 8
        date_col = int(request.form["date_column"])-1
        value_col = int(request.form["value_column"])-1

        data.iloc[:,date_col] = pandas.to_datetime(data.iloc[:,date_col])
        data_train = data.iloc[:-8,value_col]
        data_test = data.iloc[-8:,value_col]
        predictions = data_test.copy()

        naive_fit = processing.seasonal_naive(data_train, seasonality, forecast_horizon)[0]
        naive_forecast = processing.seasonal_naive(data_train, seasonality, forecast_horizon)[1]
        naive_residuals = data_train - naive_fit
        naive_residual_checks = processing.residual_checks(naive_residuals.dropna(), seasonality)



        img = processing.plot_scatter(data.iloc[:, 0], data.iloc[:, 1])
        img2 = processing.plot_scatter(naive_fit.index.array, naive_fit)

        return render_template('view.html',
                               tab=data.to_html(classes='onlyone'),
                               image=img.decode('utf8'),
                               image2=img2.decode('utf8'),
                               tab2=naive_residual_checks.to_html(classes='onlyone',index=False)
                               )

    return '''
        <html>
            <body>
                <p>Select the file that you want to calculate:</p>
                <form method="post" action="/forecast" enctype="multipart/form-data">
                    <p><input type="file" name="input"/></p>
                    <p>Date column number: <input name="date_column"/></p>
                    <p>Value column number: <input name="value_column"/></p>
                    <p><input type="submit" value="Submit"/></>
                </form>
            </body>
        </html>
    '''




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







































