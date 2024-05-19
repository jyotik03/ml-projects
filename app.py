from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app = application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        writing_score_str = request.form.get('writing score')
        reading_score_str = request.form.get('reading score')

        if writing_score_str is None or reading_score_str is None:
            # Handle the case where the required fields are missing
            return render_template('home.html', error_message="Please provide values for both writing score and reading score.")

        try:
            writing_score = float(writing_score_str)
            reading_score = float(reading_score_str)
        except ValueError:
            # Handle the case where the provided values cannot be converted to floats
            return render_template('home.html', error_message="Invalid input. Please provide valid numeric values for writing score and reading score.")

        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=reading_score,
            writing_score=writing_score
        ) 

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])

    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)