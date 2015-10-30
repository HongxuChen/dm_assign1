#!/usr/bin/env python
import logging
import logging.config

import yaml

logging_yaml = 'log.yaml'

label_dict = {
    0: 'age',
    1: 'height',
    2: 'weight',
    3: 'bmi',
    4: 'gender',
    5: 'race'
}


def init_logger():
    with open(logging_yaml) as f:
        data = yaml.load(f)
    logging.config.dictConfig(data)


def get_logger():
    return logging.getLogger('')
