import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the data from spam-data.csv
data = pd.read_csv("spam-data.csv")

# Load the data from spam-data.csv
data = pd.read_csv("spam-data.csv")

# Print the column names
print(data.columns)

# Print the first few rows of the DataFrame
print(data.head())

# Separate features (X) and target (y)
X = data.drop(columns=["Spam"])
y = data["Spam"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Load the first email from emails.txt
with open("emails.txt", "r") as file:
    first_email = file.readline()

# Test the first email for spam using the trained model
email_features = [len(first_email)]  # Example feature extraction, can be extended
is_spam = model.predict([email_features])[0]

# Print the result
if is_spam == 1:
    print("The first email is classified as spam.")
else:
    print("The first email is not classified as spam.")


print('task is successful')