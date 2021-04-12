from sklearn.datasets import fetch_california_housing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression


class_df = pd.read_csv("animal_classes.csv")
animals_train = pd.read_csv("animals_train.csv")
animals_test = pd.read_csv("animals_test.csv")
animal_class = class_df.loc[:, ["Class_Number", "Class_Type", "Animal_Names"]]
animal_class.index += 1


animal_number = animal_class["Class_Number"].to_list()
animal_type = animal_class["Class_Type"].to_list()
animals_train_number = animals_train["class_number"].to_list()

target = [
    animal_class["Class_Type"][x] for x in animals_train_number if x in animal_number
]


x_train, x_test, y_train, y_test = train_test_split(
    target, animals_test, random_state=11
)

knn = KNeighborsClassifier()
knn.fit(X=x_train, y=y_train)
predicted = knn.predict(X=x_test)
expected = y_test
print(predicted)
print(expected)
