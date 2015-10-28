#!/usr/bin/env python
import logging
import sys
import yaml
import logging.config

logging_yaml = 'log.yaml'


def init_logger():
    with open(logging_yaml) as f:
        data = yaml.load(f)
    logging.config.dictConfig(data)


def get_logger():
    return logging.getLogger('')
