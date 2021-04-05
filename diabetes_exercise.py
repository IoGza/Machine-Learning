""" Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line """

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets


# how many sameples and How many features?
db = datasets.load_diabetes()
print(db.data.shape)

"There are 442 total samples with 10 features"


# What does feature s6 represent?
print(db.DESCR)
"s6 represents the glucose blood sugar level"


# print out the coefficient
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(db.data, db.target, random_state=11)

linear_regression = LinearRegression()

linear_regression.fit(X_train, y_train)

print(linear_regression.coef_)


# print out the intercept
print(linear_regression.intercept_)

# create a scatterplot with regression line
predicted = linear_regression.predict(X_test)
expected = y_test
plt.plot(expected, predicted, ".")

x = np.linspace(0, 330, 100)
y = x
plt.plot(x, y)
plt.show()