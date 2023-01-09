# Random-Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd 

# Load the dataset
X = pd.read_csv("Breast_cancer_data.csv")
y = X.pop('diagnosis')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = RandomForestClassifier()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: ", accuracy)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

# true labels
y_true = [1, 0, 1, 0, 1, 0]

# predicted labels
y_pred = [0, 1, 1, 0, 0, 1]

# compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# plot confusion matrix
plt.imshow(conf_matrix, cmap='binary', interpolation='None')
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


confusion_matrix = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'])
print("\nConfusion matrix\n")
print("\n")


print(confusion_matrix)
print("\n")
precision = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])
print("\n")
print("\nPRECISION:\n")
print(precision)

print("\n")
recall = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])
print("\nRECALL:\n")
print(recall)
recall = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])
print("\n")
print("\nF1_SCORE:\n")
f1_score = 2 * (precision * recall) / (precision + recall)
print(f1_score)
