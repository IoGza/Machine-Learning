from sklearn.datasets import load_digits

digits = load_digits()

# print(digits.DESCR)
"""
print(digits.data[13])
print(digits.data.shape)

# prints the target value for row 13, in this case, it's the number 3
print(digits.target[13])

# shows simply the answer for the target row
print(digits.target.shape)

print(digits.images[13])

import matplotlib.pyplot as plt

figure, axes = plt.subplots(nrows=4,ncols=6,figsize=(6,4))
# Python zip func bundles the 3 iterables and produces one iterable
for item in zip(axes.ravel(),digits.images,digits.target):
    axes, image, target = item
    # displays multichannel (RGB) or signle-channel ("grayscale") image data
    axes.imshow(image,cmap=plt.cm.gray_r)
    # remove x-axis tick marks
    axes.set_xticks([])
    # remove y-axis tick marks
    axes.set_yticks([])
    # the target value of the image
    axes.set_title(target)

plt.tight_layout()
plt.show()
"""


from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    digits.data, digits.target, random_state=11
)

print(data_train.shape)

print(target_train.shape)

print(data_test.shape)

print(target_test.shape)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
# Load the training data into the model using the fit method
# Note: the KNeighborsClassifer fit method does not do calculations
# It just loads the model

knn.fit(X=data_train, y=target_train)
# Returns an array containing the predicted class of each test image:
# creates an array of digits

predicted = knn.predict(X=data_test)

expected = target_test

print(predicted[:20])
print(expected[:20])

wrong = [(p, e) for (p, e) in zip(predicted, expected) if p != e]

print(wrong)

print(format(knn.score(data_test, target_test), ".2%"))

from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_true=expected, y_pred=predicted)

print(confusion)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt2

confusion_df = pd.DataFrame(confusion, index=range(10), columns=range(10))

figure = plt2.figure(figsize=(7, 6))
axes = sns.heatmap(confusion_df, annot=True, cmap=plt2.cm.nipy_spectral_r)
plt2.show()

print("done")