import requests
from sklearn import datasets
from sklearn import linear_model
import time

IRIS = datasets.load_iris()
FEATURES = IRIS.data.tolist()
CLASSES = IRIS.target.tolist()


def expected_classes(penalty, solver):
    model = linear_model.LogisticRegression(penalty=penalty, solver=solver)
    model.fit(FEATURES, CLASSES)
    return model.predict(FEATURES)


def run_model(penalty, solver):
    columns = [col.replace(' ', '_').replace('_(cm)', '') for col in IRIS.feature_names]
    train_data = {
        'columns': columns,
        'features': FEATURES,
        'classes': CLASSES,
        'penalty': penalty,
        'solver': solver
    }
    response = requests.post('http://api:5000/models', json=train_data)
    model_id = response.json()['model_id']
    assert(response.status_code == 201)
    assert(model_id is not None)

    predict_data = {'sepal_length': 1.5, 'sepal_width': 0.4, 'petal_length': 0.9, 'petal_width': 0.2}
    response = requests.get('http://api:5000/models/%s' % model_id, data=predict_data)
    while 'model_status' in response.json().keys():
        print('waiting')
        time.sleep(1)
        response = requests.get('http://api:5000/models/%s' % model_id, data=predict_data)

    predicted = []
    for feature in FEATURES:
        predict_data = {col: feature[i] for i, col in enumerate(columns)}
        response = requests.get('http://api:5000/models/%s' % model_id, data=predict_data)
        print(response.json())
        predicted.append(response.json()['predicted_class'])

    for predicted, expected in zip(predicted, expected_classes(penalty, solver)):
        assert(predicted == int(expected))


def test_models():
    run_model('l1', 'liblinear')
    run_model('l2', 'liblinear')
