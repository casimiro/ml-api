from celery import Celery
from flask import Flask
from config import logging
from webargs import fields
from webargs.flaskparser import use_kwargs
from sklearn import linear_model


app = Flask(__name__)
app_queue = Celery(__name__, broker='redis://redis:6379/0')

train_args = {
    'columns': fields.List(fields.Str(), required=True),
    'features': fields.List(fields.List(fields.Float()), required=True),
    'classes': fields.List(fields.Str(), required=True)
}

predict_args = {
    'sepal_length': fields.Float(required=True),
    'sepal_width': fields.Float(required=True),
    'petal_length': fields.Float(required=True),
    'petal_width': fields.Float(required=True)
}


@app_queue.task
def training(columns, features, classes):
    model = linear_model.LogisticRegression()
    model.fit(features, classes)
    pass


@app.route('/models', methods=['POST'])
@use_kwargs(train_args)
def train(columns, features, classes):
    return 'called post method'


@app.route('/models/<int:model_id>', methods=['GET'])
@use_kwargs(predict_args)
def predict(sepal_length, sepal_width, petal_length, petal_width, model_id):
    logging.info(model_id)
    logging.info(sepal_length)
    return 'called get method'
