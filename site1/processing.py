import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import statistics

def do_calculation (num1,num2):
    return num1 + num2

def calculate_mean(number_list):
    try:
        return "The average of the numbers is {result}".format(result=statistics.mean(number_list))
    except statistics.StatisticsError as exc:
        return "Error calculating average: {}".format(exc)


def process_data(input_data):
    result = ""
    for line in input_data.splitlines():
        if line != "":
            numbers = [float(n.strip()) for n in line.split(",")]
            result += str(sum(numbers))
        result += "\n"
    return result

def plot_scatter(x,y):
    plt.clf()
    plt.scatter(x, y, alpha=0.5)

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_png = base64.b64encode(img.getvalue())
    return img_png