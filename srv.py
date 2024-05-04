from flask import Flask, request, jsonify,render_template
from flask_cors import CORS  # Import the CORS extension
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
CORS(app)
# Load the dataset and preprocess as needed
dataset_path = "D:\\datasets\\boston_housing\\HousingData.csv"
housing_data = pd.read_csv(dataset_path)
housing_data = housing_data.dropna(axis=0, how='any')

# Selecting relevant features
selected_features = ['LSTAT', 'RM', 'PTRATIO', 'INDUS', 'TAX', 'MEDV']
data_subset = housing_data[selected_features]

# Splitting the data
x = data_subset[['LSTAT', 'RM', 'PTRATIO', 'INDUS', 'TAX']]
y = data_subset['MEDV']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Training the model (you may want to consider loading a pre-trained model)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
rf=RandomForestRegressor()
rf.fit(x_train,y_train)

# Save the trained model to a file
model_filename = 'LinearRegression_trained_model.joblib'
joblib.dump(regressor, model_filename)
model_filename1 = 'RandomForest_trained_model.joblib'
joblib.dump(rf, model_filename1)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict_price', methods=['POST'])
def predict_price():
    data = request.get_json()

    # Extracting features from the request
    features = [
        data['LSTAT'],
        data['RM'],
        data['PTRATIO'],
        data['INDUS'],
        data['TAX']
    ]

    # Load the trained model random forest
    random_regressor = joblib.load(model_filename1)
    # load the trained model linear regression
    linear_regressor= joblib.load(model_filename)
    # Making predictions
    random_forest_prediction = random_regressor.predict([features])
    linear_regressor_prediction = linear_regressor.predict([features])
    return jsonify({'rfp': random_forest_prediction.tolist(),'lrp':linear_regressor_prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
