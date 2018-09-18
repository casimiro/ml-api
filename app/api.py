from celery import Celery
from flask import abort
from flask import Flask
from flask import jsonify
import json
from logging.config import dictConfig
import numpy as np
import pickle
import redis
from sklearn import linear_model
import uuid
from webargs import fields
from webargs.flaskparser import use_args
from webargs.flaskparser import use_kwargs

dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': True
        },
    }
})

flask_app = Flask(__name__)
app = Celery(__name__, backend='redis://redis', broker='redis://redis:6379/0')
app.conf.update(
    task_serializer='pickle',
    accept_content=['pickle'],
    result_serializer='pickle',
)
redis_instance = redis.Redis(host='redis')

train_args = {
    'columns': fields.List(fields.Str(), required=True),
    'features': fields.List(fields.List(fields.Float()), required=True),
    'classes': fields.List(fields.Int(), required=True),
    'penalty': fields.Str(required=True),
    'solver': fields.Str(required=True),
}

predict_args = {
    'sepal_length': fields.Float(required=True),
    'sepal_width': fields.Float(required=True),
    'petal_length': fields.Float(required=True),
    'petal_width': fields.Float(required=True)
}


@app.task
def training(features, classes, columns, penalty, solver, model_id):
    model = linear_model.LogisticRegression(penalty=penalty, solver=solver)
    model.fit(features, classes)
    data = {'model': model, 'positions': columns}
    redis_instance.set(model_id.int, pickle.dumps(data))
    return data


@flask_app.route('/models', methods=['POST'])
@use_kwargs(train_args)
def train(columns, features, classes, penalty, solver):
    flask_app.logger.info('about to train on dataset of %d rows', len(features))
    model_id = uuid.uuid1()
    redis_instance.set(model_id.int, 'training')
    training.delay(features, classes, columns, penalty, solver, model_id)
    return jsonify({'model_id': str(model_id)}), 201


@flask_app.route('/models/<uuid:model_id>', methods=['DELETE'])
def delete_model(model_id):
    flask_app.logger.info('about to delete model %s', model_id)
    id = uuid.UUID(str(model_id))
    if redis_instance.delete(id.int) == 0:
        abort(404)
    else:
        return jsonify({'status': 'model deleted'})


@flask_app.route('/models/<uuid:model_id>', methods=['GET'])
@use_args(predict_args)
def predict(args, model_id):
    flask_app.logger.info('predicting on %s', json.dumps(args))
    id = uuid.UUID(str(model_id))
    model_status = redis_instance.get(id.int)

    if model_status is None:
        abort(404)

    if model_status == b'training':
        return jsonify({'model_status': 'training'})
    else:
        model_data = pickle.loads(model_status)
        features = np.array([args[column] for column in model_data['positions']])
        predicted_class = model_data['model'].predict(features.reshape(1, len(features)))
        return jsonify({'predicted_class': int(predicted_class[0])})
