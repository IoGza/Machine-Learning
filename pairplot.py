import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

california = fetch_california_housing()


# print(california.data.shape)
# print(california.target.shape)
# print(california.feature_names)

pd.set_option("max_columns", 9)
pd.set_option("precision", 4)
pd.set_option("display.width", None)

# Create initial dataframe using california.data
cali_df = pd.DataFrame(california.data, columns=california.feature_names)
# cali_df['MedHouseValue'] = [california.target[i] for i in california.feature_names]

# Adds a columns for the median house values to the data frame
cali_df["MedHouseValue"] = pd.Series(california.target)


sns.set(font_scale=2)
sns.set_style("whitegrid")
grid = sns.pairplot(
    data=cali_df, vars=cali_df.columns[0:4]
)

plt.show()