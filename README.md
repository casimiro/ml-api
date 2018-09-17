# ML API example

Running the API should be as simple as running `docker-compose up` inside the directory containing
the docker-compose.yml file.

## Training

Training a model can be done with the following request
`curl -v http://localhost:8000/models -d @train.json --header "Content-Type: application/json`

The request will return an UUID that refers to the model being trained.

## Prediction

Getting model's prediction can be done using a GET request like this:
`curl -v http://localhost:8000/models/e0526970-baaa-11e8-9a28-0242ac150004\?sepal_length\=5.9\&sepal_width\=3.0\&petal_length\=5.1\&petal_width\=1.8`
Where `e0526970-baaa-11e8-9a28-0242ac150004` is the model UUID.
