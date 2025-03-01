import pickle
from flask import Flask, request, render_template
import logging  # Import logging module

app = Flask(__name__)

# Load the models
ridge_model = pickle.load(open("Models/ridge.pkl", "rb"))
standard_scaler = pickle.load(open("Models/scaler.pkl", "rb"))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    logging.basicConfig(level=logging.DEBUG)  # Set up logging to DEBUG level
    logging.debug("Received request for prediction")  # Log the request

    if request.method == 'POST':
        result = None  # Initialize result variable
        try:
            Temperature = float(request.form.get('Temperature')) if request.form.get('Temperature') else 0
            logging.debug(f"Temperature: {Temperature}")  # Log the temperature value

            RH = float(request.form.get('RH')) if request.form.get('RH') else 0
            logging.debug(f"Relative Humidity (RH): {RH}")  # Log the RH value

            Ws = float(request.form.get('Ws')) if request.form.get('Ws') else 0
            logging.debug(f"Wind Speed (Ws): {Ws}")  # Log the wind speed value

            Rain = float(request.form.get('Rain')) if request.form.get('Rain') else 0
            logging.debug(f"Rain: {Rain}")  # Log the rain value

            FFMC = float(request.form.get('FFMC')) if request.form.get('FFMC') else 0
            logging.debug(f"FFMC: {FFMC}")  # Log the FFMC value

            DMC = float(request.form.get('DMC')) if request.form.get('DMC') else 0
            logging.debug(f"DMC: {DMC}")  # Log the DMC value

            ISI = float(request.form.get('ISI')) if request.form.get('ISI') else 0
            logging.debug(f"ISI: {ISI}")  # Log the ISI value

            Classes = float(request.form.get('Classes')) if request.form.get('Classes') else 0
            logging.debug(f"Classes: {Classes}")  # Log the Classes value

            Region = float(request.form.get('Region')) if request.form.get('Region') else 0
            logging.debug(f"Region: {Region}")  # Log the Region value

            new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            result = ridge_model.predict(new_data_scaled) if new_data_scaled is not None else None

            return render_template('home.html', result=result[0] if result is not None else "No prediction available")  # Return the prediction result
        except Exception as e:
            logging.error(f"Error occurred: {str(e)}")  # Log the error
            return render_template('home.html', error=str(e))  # Handle errors and return to home

    return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
