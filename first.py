import pandas as pd
import quandl
import math

#python doesn't actually have arrays, numpy lets you use arrays
import numpy as np
import sklearn
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]




forecast_col = 'Adj. Close'
#in machine learning, you can't use NaNs
df.fillna(-99999, inplace=True)


#using int, because math.ceil returns a float
forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
#print df

#put features and labels into arrays - each into new array
X = np.array(df.drop(['label'],1))
y = np.array(df['label'])
#scaling X before putting it through classifier
X = preprocessing.scale(X)


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#clf = svm.SVR()
clf= LinearRegression()
#fit is synonymous with train
clf.fit(X_train, y_train)
#score is synonymous with test
#accuracy = error^2 for linear regression - accuracy and confidence referred to as different values
accuracy = clf.score(X_test, y_test)

print accuracy

