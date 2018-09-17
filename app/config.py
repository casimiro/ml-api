import logging


root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("{0}/{1}.log".format('/src', 'api'))
console_handler = logging.StreamHandler()
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)
