from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Load the trained model
model2 = pickle.load(open('model.pkl', 'rb'))

standard_to = StandardScaler()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Route to the dashboard page
@app.route('/dashboard', methods=['GET'])
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Fat = float(request.form['Item_Fat_Content'])
        Item_type = float(request.form['Item_Type'])
        Location = float(request.form['Outlet_Location_Type'])
        Outlet_type = float(request.form['Outlet_Type'])
        Age = float(request.form['Age_Outlet'])
        Price = float(request.form['Item_MRP'])

        # Make prediction
        prediction = model2.predict([[Fat, 0.0539, Item_type, Price, Location, Outlet_type, Age]])
        output = "{:.2f}".format(prediction[0])

        # Render index.html with prediction result
        return render_template('index.html', prediction_text=f"Predicted Sale: ₹{output}")
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
