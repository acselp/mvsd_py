import numpy
from flask import Flask, request, Response
import pandas as pd

from file_provider import FileProvider
from StudentSuccessPrediction import StudentSuccessPrediction, get_train, get_test

app = Flask(__name__)
file_provider = FileProvider()
predictor = StudentSuccessPrediction()

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json()['model']
    feature_names = list(data.keys())

    x_predict = numpy.array([float(data[f]) for f in feature_names])
    return f'{predictor.predict([x_predict])[0]}'


@app.route("/getAllTrain")
def get_all_train():
    data = pd.DataFrame(get_train())

    return Response(data.to_json(orient='records'), 'application/json')


@app.route("/getAllTest")
def get_all_test():
    data = pd.DataFrame(get_test())

    return Response(data.to_json(orient='records'), 'application/json')


@app.route("/getScore")
def get_score():
    return f'{predictor.get_score()}'