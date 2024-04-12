import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# Step 1: Load the data
data = pd.read_csv("spam-data.txt")

# Step 2: Build the logistic regression model and train it on these data
X = data.drop(columns=["Spam"])
y = data["Spam"]
model = LogisticRegression()
model.fit(X, y)

# Step 3: Locate the emails.txt file and parse it to extract the email features using Python
with open("emails.txt", "r") as file:
    emails = file.readlines()

# Step 4: Check those emails for spam and print the results on the console
vectorizer = CountVectorizer()
X_emails = vectorizer.fit_transform(emails)
y_emails = model.predict(X_emails)

for i, email in enumerate(emails):
    if y_emails[i] == 1:
        print(f"Email {i + 1}: SPAM")
    else:
        print(f"Email {i + 1}: NOT SPAM")