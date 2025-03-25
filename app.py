from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
Bootstrap(app)

# Load and train model once (instead of inside predict)
df = pd.read_csv("data2.csv")
df_data = df[["class", "comments"]]
df_x = df_data["comments"]
df_y = df_data["class"]

# Create CountVectorizer and train the model
cv = CountVectorizer()
X = cv.fit_transform(df_x)
X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.3, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template("home.html", prediction=None, user_comment="")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        return render_template('home.html', prediction=my_prediction, user_comment=comment)

if __name__ == '__main__':
    app.run(debug=True)
