import numpy as np
from flask import Flask, request, render_template
import pickle
import traceback

app = Flask(__name__, static_folder='Static')

try:
    model = pickle.load(open('BP_model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        int_features = [float(x) for x in request.form.values()]
        features = [np.array(int_features)]
        
        if model is None:
            return render_template('index.html', prediction_text='Error: Model not loaded')
        
        prediction = model.predict(features)
        output = round(prediction[0], 2)
        
        return render_template('index.html', prediction_text='Predicted Blood Pressure is {}'.format(output))
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        print(traceback.format_exc())
        return render_template('index.html', prediction_text='Error in prediction: {}'.format(str(e)))

if __name__ == "__main__":
    app.run(debug=True)