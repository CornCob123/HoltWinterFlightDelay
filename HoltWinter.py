import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from statsforecast import StatsForecast
from statsforecast.models import HoltWinters
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv('C:\\Users\\Desktop\\airport_delay_new.csv', 
                   parse_dates=['FlightDate']).loc[:, ['FlightDate', 'DestCityName', 'ArrDel15']]
data = data.groupby(['FlightDate', 'DestCityName'], as_index=False)['ArrDel15'].count()
data = data.rename(columns={'FlightDate': 'FlightDate','DestCityName': 'DestCityName', 'ArrDel15': 'NumberOfDelayedFlights'})
data = data.groupby('DestCityName', as_index=False).get_group('Miami, FL')
data = data.sort_values(by=['FlightDate'])

data['FlightDate'] = pd.to_datetime(data['FlightDate']) 
train = data.loc[(data['FlightDate'] >= '2015-02-01') & (data['FlightDate'] < '2015-02-25')]
test = data.loc[(data['FlightDate'] >= '2015-02-24') & (data['FlightDate'] <= '2015-02-28')]

train = train.sort_values(by=['FlightDate'])
test = test.sort_values(by=['FlightDate'])

train.to_csv('C:\\Users\\Desktop\\dataAnalysisPt2\\trainWinter.csv',index=False)
test.to_csv('C:\\Users\\Desktop\\dataAnalysisPt2\\testWinter.csv',index=False)

print(train)
train['NumberOfDelayedFlights'] = train['NumberOfDelayedFlights'].replace([0], 0.01)
train['NumberOfDelayedFlights'].fillna(value = 0.01, inplace = True)
test['NumberOfDelayedFlights'] = test['NumberOfDelayedFlights'].replace([0], 0.01)
test['NumberOfDelayedFlights'].fillna(value = 0.01, inplace = True)

hwmodel = ExponentialSmoothing(train.NumberOfDelayedFlights,trend='add',seasonal='mul',seasonal_periods=7).fit()
test_pred = hwmodel.forecast(5)
test_pred.columns =['FlightDate', 'NumberOfDelayedFlights']

ax = train.plot(x='FlightDate', y='NumberOfDelayedFlights', title='Miami Flights Train Data')
ax.set_xlabel("FlightDate")
ax.set_ylabel("NumberOfDelayedFlights")
plt.show()
ax = test.plot(x='FlightDate', y='NumberOfDelayedFlights', title='Miami Flights Test Data')
ax.set_xlabel("FlightDate")
ax.set_ylabel("NumberOfDelayedFlights")
plt.show()
ax = test_pred.plot(x='FlightDate', y='NumberOfDelayedFlights', title='Miami Flights Prediction')
ax.set_xlabel("FlightDate")
ax.set_ylabel("NumberOfDelayedFlights")
plt.show()
print(test_pred)

x1 = train['FlightDate']
x2 = test['FlightDate']
y1 = train['NumberOfDelayedFlights']
y2 = test['NumberOfDelayedFlights']

plt.plot(x1, y1)
plt.plot(x2, y2)
plt.xlabel("Flight Date")
plt.ylabel("NumberOfDelayedFlights")
plt.legend() 

plt.show()

print(np.sqrt(mean_squared_error(test,test_pred)))

