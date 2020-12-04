
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, request, session, make_response, render_template
import processing
import pandas as pd

app = Flask(__name__)
app.config["DEBUG"] = True
app.config["SECRET_KEY"] = 'warfhjwoiuj348'


@app.route("/forecast", methods=["GET","POST"])
def forecast():

    if request.method =="POST":
        data = pd.read_csv(request.files.get('input'))
        col_number = int(request.form["date_column"])

        data.iloc[:,col_number] = pd.to_datetime(data.iloc[:,col_number])

        img = processing.plot_scatter(data.iloc[:,0], data.iloc[:,1])

        return render_template('view.html', tab=data.to_html(classes='onlyone'), image=img.decode('utf8'),number=col_number)

    return '''
        <html>
            <body>
                <p>Select the file that you want to calculate:</p>
                <form method="post" action="/forecast" enctype="multipart/form-data">
                    <p><input type="file" name="input"/></p>
                    <p><input name="date_column" value="Input the column number for date column"/></p>
                    <p><input type="submit" value="Submit"/></>
                </form>
            </body>
        </html>
    '''




@app.route("/", methods=["GET","POST"])
def intput_output_page():
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
                <form method="post" action="." enctype="multipart/form-data">
                    <p><input type="file" name="submission"/></p>
                    <p><input type="submit" value="Process the file"/></>
                </form>
            </body>
        </html>
    '''








@app.route("/list2", methods=["GET","POST"])
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



        if request.form["action"] == "Do calculation":
            result = processing.calculate_mean(session["inputs"])
            session["inputs"].clear() # how does inputs clearing solve the reset issue?
            session.modified = True
            return '''
                <html>
                    <body>
                        <p>{result}</p>
                        <p><a href="/list2">Click here to start again</a>
                    </body>
                </html>
            '''.format(result=result)

    return '''
        <html>
            <body>
                {previous}
                {errors}
                <p>Enter your number:</p>
                <form method="post" action="/list2">
                    <p><input name="user_input" /></p>
                    <p><input type="submit" name="action" value="Add number" /></p>
                    <p><input type="submit" name="action" value="Do calculation" /></p>
                </form>
            </body>
        </html>
    '''.format(errors=errors,previous=number_list)


















@app.route("/list", methods=["GET", "POST"])
def hello_world():

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
                <form method="post" action="/list">
                    <p><input name="numberA" /></p>
                    <p><input name="number2" /></p>
                    <p><input type="submit" value="Do calculation" /></p>
                </form>
            </body>
        </html>
    '''.format(errors=errors)






































