from __future__ import absolute_import
from models.make_model import make_model

__factory = {
    'make_model':make_model,
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))

    m = __factory[name](*args, **kwargs)

    return m
