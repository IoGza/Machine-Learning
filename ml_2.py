import pandas as pd

nyc = pd.read_csv("ave_hi_nyc_jan_1895-2018.csv")

# print(nyc.head(3))

print(nyc.Date.values)

# -1 tells the reshape function to create as many rows as needed
print(nyc.Date.values.reshape(-1, 1))

print(nyc.Temperature.values)

from sklearn.model_selection import train_test_split

# 75% of the data goes to training, 25% goes to testing the data
X_train, X_test, y_train, y_test = train_test_split(
    nyc.Date.values.reshape(-1, 1), nyc.Temperature.values, random_state=11
)

# The data that is being fed to the train_test_split function
print(X_train.shape)
print(X_test.shape)

# The target data used
print(y_train.shape)
# Used to compare what is produced from the xtest
print(y_test.shape)

from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()

# The fit method expects the samples and the targets for training
linear_regression.fit(X=X_train, y=y_train)

print(linear_regression.coef_)

print(linear_regression.intercept_)


predicted = linear_regression.predict(X_test)

expected = y_test

for p, e in zip(predicted[::5], expected[::5]):  # Check every 5th
    print(f"predicted: {p:.2f},expected: {e:.2f}")


# Lambda implements y = mx + b
predict = lambda x: linear_regression.coef_ * x + linear_regression.intercept_

print(predict(2021))
print(predict(1899))


# Vizualize the data
# Create a scatter plot with regression line
# cooler temps shown in darker colors
import seaborn as sns

axes = sns.scatterplot(
    data=nyc,
    x="Date",
    y="Temperature",
    hue="Temperature",
    palette="winter",
    legend=False,
)

axes.set_ylim(10, 70)  # scale y-axis

import numpy as np

# An array of the lowest and hughest dates
x = np.array([min(nyc.Date.values), max(nyc.Date.values)])
print(x)
# The corresponding temperature for the dates
y = predict(x)
print(y)

import matplotlib.pyplot as plt

line = plt.plot(x, y)
plt.show()