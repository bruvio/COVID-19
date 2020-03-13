import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import warnings
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")


nazionale_df = pd.read_csv("dpc-covid19-ita-andamento-nazionale.csv",parse_dates=['data'],
                                index_col=['data'])

print(nazionale_df.head(5))
print(nazionale_df.columns)
totale_casi = np.array(nazionale_df['totale_casi'])
casi_attivi = totale_casi - nazionale_df['dimessi_guariti']

data =  nazionale_df.index.values
# convert date to unix time for fit use
dates = pd.to_numeric(data)

# totale_casi_fit = np.polyfit(dates, np.log(totale_casi), 1)



def func(x, a, b, c): # Hill sigmoidal equation from zunzun.com
    return  a * np.power(x, b) / (np.power(c, b) + np.power(x, b))

def func1(x, a, b, c): # Hill sigmoidal equation from zunzun.com
    return  a * np.power(x, b) / (np.power(c, b) + np.power(x, b))

def func2(x, a, b,c,d):
        return d + ((a-d)/(1+(x/c)**b))

def exp_func(x, a, b ):
    return a*np.exp(b*x)
def log_func(t,a,b):
    return a+b*np.log(t)

# using an array of integer to calculate fit to avoid overflow
x = np.arange(0,len(totale_casi))

popt1, pcov1 = curve_fit(exp_func,  x,  totale_casi)
popt2, pcov2 = curve_fit(exp_func,  x,  casi_attivi)



# plt.figure()
# plt.plot(x, totale_casi, 'ko', label="Original Data")
# plt.plot(x, exp_func(x, *popt1), 'r-', label="Fitted Curve")
#
#
# plt.legend()
# plt.show()

plt.figure()
plt.plot(nazionale_df.index.values,nazionale_df['totale_casi'],'k',label='totale casi',marker='x')
plt.plot(nazionale_df.index.values,casi_attivi,'b',label='casi attivi')
plt.plot(nazionale_df.index.values, exp_func(x, *popt1), 'r-', label="Fitted Curve - casi totali")
plt.plot(nazionale_df.index.values, exp_func(x, *popt2), 'm-', label="Fitted Curve - casi attivi")

plt.plot(nazionale_df.index.values,nazionale_df['isolamento_domiciliare'],label='isolamento_domiciliare',marker='>')
plt.plot(nazionale_df.index.values,nazionale_df['deceduti'],label='deceduti',marker='o')
plt.plot(nazionale_df.index.values,nazionale_df['dimessi_guariti'],label='dimessi_guariti',marker='s')
# plt.plot(date, totale_casi_fit,label='casi totali - fit', marker='^')
plt.xticks(rotation=15, ha="right")
plt.legend(loc='best')

# plt.show()
# print(popt,pcov)



x = np.arange(0, len(totale_casi))

try:
    fittedParameters, pcov = curve_fit(func1, x, totale_casi, maxfev=5000)
    modelPredictions = func1(x, *fittedParameters)

    absError = modelPredictions - totale_casi

    SE = np.square(absError)  # squared errors
    MSE = np.mean(SE)  # mean squared errors
    RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (np.var(absError) / np.var(totale_casi))

    print('Parameters:', fittedParameters)
    print('RMSE:', RMSE)
    print('R-squared:', Rsquared)
    # create data for the fitted equation plot
    xModel = np.linspace(min(x), max(x))
    yModel = func1(xModel, *fittedParameters)

    # first the raw data as a scatter plot
    plt.figure()
    length = len(totale_casi)
    plt.plot(x, totale_casi, 'ko', label="Original Data")
    # plt.plot(nazionale_df.index.values, totale_casi, label="Original Data - " + country + ' - ' + label)
    # plt.figure()
    # now the model as a line plot
    plt.plot(xModel, yModel, label="Fitted Curve - casi totali")

    plt.xticks(rotation=15, ha="right")
    # plt.xticks(dates)
    # plt.major_formatter(mdates.DateFormatter("%m-%d"))
    # plt.minor_formatter(mdates.DateFormatter("%m-%d"))
    plt.legend(loc='best')

    plt.show()
except ValueError:
    logger.error('failed to fit data')


