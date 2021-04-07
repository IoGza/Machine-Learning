from sklearn.datasets import fetch_california_housing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression


california = fetch_california_housing()

# print(california.DESCR)
# print(california.data.shape)
# print(california.target.shape)
# print(california.feature_names)

pd.set_option("precision", 4)
pd.set_option("max_columns", 9)
pd.set_option("display.width", None)

cali_df = pd.DataFrame(california.data, columns=california.feature_names)
cali_df["MedHouseValue"] = pd.Series(california.target)

# print(cali_df.head()) #look at first 5 rows
# print(cali_df.describe())

sample_df = cali_df.sample(frac=0.1, random_state=17)

sns.set(font_scale=2)
sns.set_style("whitegrid")

for feature in california.feature_names:
    plt.figure(figsize=(8, 4.5))
    sns.scatterplot(
        data=sample_df,
        x=feature,
        y="MedHouseValue",
        hue="MedHouseValue",
        palette="cool",
        legend=False,
    )

# plt.show()

x_train, x_test, y_train, y_test = train_test_split(
    california.data, california.target, random_state=11
)


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

linear_regression = LinearRegression()
linear_regression.fit(X=x_train, y=y_train)

for i, name in enumerate(california.feature_names):
    print(f"{name}: {linear_regression.coef_[i]}")

predicted = linear_regression.predict(x_test)
print(predicted[:5])

expected = y_test
print(expected[:5])

df = pd.DataFrame()

df["Expected"] = pd.Series(expected)
df["Predicted"] = pd.Series(predicted)

print(df[:10])

import matplotlib.pyplot as plt2

figure = plt2.figure(figsize=(9,9))
axes = sns.scatterplot(
    data=df,x="Expected",y="Predicted",hue="Predicted",palette="cool",
    legend = False
)

start = min(expected.min(), predicted.min())
print(start)

end = max(expected.max(),predicted.max())
print(end)

axes.set_xlim(start,end)
axes.set_ylim(start,end)

line = plt2.plot([start,end], [start,end], "k--")
plt2.show()