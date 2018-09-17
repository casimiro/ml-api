from celery import Celery
from config import logging
from flask import abort
from flask import Flask
from flask import jsonify
import numpy as np
import pickle
import redis
from sklearn import linear_model
import uuid
from webargs import fields
from webargs.flaskparser import use_args
from webargs.flaskparser import use_kwargs


app = Flask(__name__)
app_queue = Celery(__name__, broker='redis://redis:6379/0')
redis_instance = redis.Redis(host='redis')

train_args = {
    'columns': fields.List(fields.Str(), required=True),
    'features': fields.List(fields.List(fields.Float()), required=True),
    'classes': fields.List(fields.Int(), required=True)
}

predict_args = {
    'sepal_length': fields.Float(required=True),
    'sepal_width': fields.Float(required=True),
    'petal_length': fields.Float(required=True),
    'petal_width': fields.Float(required=True)
}


@app_queue.task
def training(features, classes, columns, model_id):
    model = linear_model.LogisticRegression()
    redis_instance.set(model_id.int, 'training')
    model.fit(features, classes)
    data = {'model': model, 'positions': columns}
    redis_instance.set(model_id.int, pickle.dumps(data))


@app.route('/models', methods=['POST'])
@use_kwargs(train_args)
def train(columns, features, classes):
    logging.info('about to train')
    model_id = uuid.uuid1()
    training(features, classes, columns, model_id)
    return jsonify({'model_id': str(model_id)}), 201


@app.route('/models/<uuid:model_id>', methods=['GET'])
@use_args(predict_args)
def predict(args, model_id):
    id = uuid.UUID(str(model_id))
    model_status = redis_instance.get(id.int)

    if model_status is None:
        abort(404)

    if model_status == b'training':
        return 'training'
    else:
        model_data = pickle.loads(model_status)
        features = np.array([args[column] for column in model_data['positions']])
        predicted_class = model_data['model'].predict(features.reshape(1, len(features)))
        logging.info(predicted_class)
        return jsonify({'predicted_class': int(predicted_class[0])})
