import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import numpy as np

nyc = pd.read_csv("ave_yearly_temp_nyc_1895-2017.csv")
print(nyc.columns.values)
nyc.columns = ["Date", "Temperature", "Anomaly"]

nyc.Date = nyc.Date.floordiv(100)

print(nyc.head(3))

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

linear_regression = LinearRegression()

linear_regression.fit(X=X_train, y=y_train)

predicted = linear_regression.predict(X_test)
expected = y_test

for p, e in zip(predicted[::5], expected[::5]):
    print(f"predicted: {p:.2f}, expected: {e:.2f}")

predict = lambda x: linear_regression.coef_ * x + linear_regression.intercept_

print(predict(2017))
print(predict(1895))

axes = sns.scatterplot(
    data=nyc,
    x="Date",
    y="Temperature",
    hue="Temperature",
    palette="winter",
    legend=False,
)

axes.set_ylim(10, 70)

x = np.array([min(nyc.Date.values), max(nyc.Date.values)])

y = predict(x)

'''Higher average temperature compared to the January data, the temperature is also much closer to 
the slope of the line and less anamolous in nature
'''
line = plt.plot(x, y)
plt.show()