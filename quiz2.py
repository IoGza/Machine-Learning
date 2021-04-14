import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from pandas import DataFrame

class_df = pd.read_csv("animal_classes.csv")
animals_train = pd.read_csv("animals_train.csv")
animals_test = pd.read_csv("animals_test.csv")
animal_class = class_df.loc[:, ["Class_Number", "Class_Type"]]
animal_class.index += 1

animal_number = animal_class["Class_Number"].to_list()
animals_train_number = animals_train["class_number"]


knn = KNeighborsClassifier()
knn.fit(X=animals_train.drop("class_number", axis=1), y=animals_train_number)
predicted = knn.predict(X=animals_test.drop("animal_name", axis=1))



class_type = [animal_class["Class_Type"][x] for x in predicted if x in animal_number]
name = animals_test["animal_name"]
class_type = DataFrame(class_type, columns=["prediction"])


animals = pd.concat([name, class_type], axis=1)
animals.to_csv("quiz_predictions.csv", index=0, header=True)