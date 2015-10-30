#!/usr/bin/env python
import logging
import logging.config

import yaml

logging_yaml = 'log.yaml'

isclf_dict = {
    0: True,
    1: True,
    2: True,
    3: True,
    4: True,
    5: True
}


def init_logger():
    with open(logging_yaml) as f:
        data = yaml.load(f)
    logging.config.dictConfig(data)


def get_logger():
    return logging.getLogger('')
