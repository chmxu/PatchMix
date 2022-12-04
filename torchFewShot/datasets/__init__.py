from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .miniImageNet import miniImageNet
from .tieredImageNet import tieredImageNet
from .miniImageNet_load import miniImageNet_load
from .cifar_fs import cifar_fs
from .FC100 import FC100
from .CUB import CUB

__imgfewshot_factory = {
        'miniImageNet': miniImageNet,
        'tieredImageNet': tieredImageNet,
        'miniImageNet_load': miniImageNet_load,
        'cifar_fs' : cifar_fs,
        'FC100' : FC100,
        'CUB' : CUB
}


def get_names():
    return list(__imgfewshot_factory.keys()) 


def init_imgfewshot_dataset(name, **kwargs):
    if name not in list(__imgfewshot_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgfewshot_factory.keys())))
    return __imgfewshot_factory[name](**kwargs)

