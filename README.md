# ML API example

Running the API should be as simple as running `docker-compose up` inside the directory containing
the docker-compose.yml file.

## Training

Training a model can be done with the following request
`curl -v http://localhost:8000/models -d @train.json --header "Content-Type: application/json`

The request will return an UUID that refers to the model being trained.

## Prediction

Getting model's prediction can be done using a GET request like this:
`curl -v http://localhost:8000/models/e0526970-baaa-11e8-9a28-0242ac150004?sepal_length=5.9&sepal_width=3.0&petal_length=5.1&petal_width=1.8`
Where `e0526970-baaa-11e8-9a28-0242ac150004` is the model UUID.

## Running the tests

After running the stack (e.g, `docker-compose up -d`) just run `compose exec api pytest`

## Notes

There's still room for improvement, and although I do not have time to make all the improvements, it's
good to have them listed here:
 - test code needs more refactoring (pytest plugin I relied on is not feature complete on Python 3.7)
 - I didn't make sure all sockets (including those used to comunicate with redis) are operating
   as non blocking sockets -- it might be needed some gevent patching.
 - of course it would be impotant to have more options (not only penalty and solver) for model creation.
   I'm assuming that the client sends us only the training data and we are not splitting this data into
   training and testing -- it might be a good a idea to do that, even though the client can do the separation
   on his/her side.
 - I'm currently using `pickle` as serialization method on celery. As pointed on Celery documentation it
   can be exploited to run undesired code on host. As the application is running in a docker container,
   that can be not a problem but it definitely deserves some investigation.  `pickle` was used in favor of
   `json` as I didn't manage to make `sklearn.linear_model.Logisticregression` json serializable. As this
   was my first time playing with `Celery` (and `kombu`) I've had a hard time trying to do that. =)
