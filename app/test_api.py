import pytest
import requests
import time
from sklearn import linear_model
from sklearn import datasets
import uuid


@pytest.fixture
def iris():
    return datasets.load_iris()


@pytest.fixture
def expected_classes(iris, penalty, solver):
    model = linear_model.LogisticRegression(penalty=penalty, solver=solver)
    model.fit(iris.data.tolist(), iris.target.tolist())
    return model.predict(iris.data.tolist())


@pytest.fixture
def treated_columns(iris):
    return [col.replace(' ', '_').replace('_(cm)', '') for col in iris.feature_names]


@pytest.fixture
def api_url():
    return 'http://api:5000/models'


@pytest.fixture
def training_data(iris, penalty, solver, treated_columns):
    return {
        'columns': treated_columns,
        'features': iris.data.tolist(),
        'classes': iris.target.tolist(),
        'penalty': penalty,
        'solver': solver
    }


@pytest.fixture
def solver():
    return 'liblinear'


@pytest.fixture
def penalty():
    return 'l1'


@pytest.fixture
def creation_response(training_data, api_url):
    return requests.post(api_url, json=training_data)


@pytest.fixture
def deletion_response(api_url, model_id):
    return requests.delete(api_url + '/%s' % model_id)


@pytest.fixture
def model_id(creation_response):
    return creation_response.json()['model_id']


def describe_model_creation_with_l1_penalty():
    def returns_201(creation_response):
        assert creation_response.status_code == 201

    def returns_model_id(creation_response):
        assert creation_response.json()['model_id'] is not None


def describe_model_creation_with_l2_penalty():
    @pytest.fixture
    def solver():
        return 'liblinear'

    @pytest.fixture
    def penalty():
        return 'l2'

    def returns_201(creation_response):
        assert creation_response.status_code == 201

    def returns_model_id(creation_response):
        assert creation_response.json()['model_id'] is not None


def describe_model_execution():
    def predicts_like_local_model(treated_columns,
                                  iris,
                                  expected_classes,
                                  api_url,
                                  model_id):
        prediction_data = {'sepal_length': 1.5, 'sepal_width': 0.4, 'petal_length': 0.9, 'petal_width': 0.2}
        response = requests.get(api_url + '/%s' % model_id, data=prediction_data)
        while 'model_status' in response.json().keys():
            print('waiting for model training to complete')
            time.sleep(1)
            response = requests.get(api_url + '/%s' % model_id, data=prediction_data)

        predicted = []
        for feature in iris.data.tolist():
            predict_data = {col: feature[i] for i, col in enumerate(treated_columns)}
            response = requests.get(api_url + '/%s' % model_id, data=predict_data)
            predicted.append(response.json()['predicted_class'])

        for predicted, expected in zip(predicted, expected_classes):
            assert predicted == int(expected)


def describe_model_deletion_with_valid_id():
    def it_returns_200(deletion_response):
        assert deletion_response.status_code == 200


def describe_model_deletion_with_invalid_id():
    @pytest.fixture
    def model_id():
        return str(uuid.uuid1())

    def it_returns_404(deletion_response):
        assert deletion_response.status_code == 404
