import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Step 1: Load the data
data = pd.read_csv("spam-data.txt")

# Step 2: Divide it into training and testing parts
X = data.drop(columns=["Spam"])
y = data["Spam"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build the regression model and train it on the training data
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Evaluate your model and print the confusion matrix
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
