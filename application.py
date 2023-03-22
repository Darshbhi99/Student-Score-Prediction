from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import predict_marks

application = Flask(__name__)
dct = {}

@application.route("/")
def index():
    return render_template("index.html")

@application.route("/predict", methods=['GET', 'POST'])
def prediction():
    if request.method == 'GET':
        return render_template("home.html")
    else:
        dct["writing_score"] = request.form['writing_score']
        dct["reading_score"] = request.form['reading_score']
        dct["gender"] = request.form["gender"]
        dct["parental_level_of_education"] = request.form["parental_level_of_education"]
        dct["race_ethnicity"] = request.form["ethnicity"]
        dct["lunch"] = request.form["lunch"]
        dct["test_preparation_course"] = request.form["test_preparation_course"]
        pred = predict_marks(dct, new_model = True)
        return render_template("home.html", results = pred[0])
    
if __name__ == "__main__":
    application.run(host="0.0.0.0")