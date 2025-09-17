# coding=utf8
from __future__ import print_function

import builtins
import logging


def get_process_name():
    import sys, os
    rank = os.environ.get("RANK", None)
    if rank:
        return "RANK_{}".format(rank)
    else:
        return ""

class GRSRecomLogger(object):
    FORMAT = '%(asctime)s,%(msecs)03d %(levelname)s [{}#%(process)d] %(filename)s:%(lineno)d - %(message)s'.format(get_process_name())
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    @classmethod
    def stdLogger(cls, name):
        logging.basicConfig(level=logging.INFO,
                            format=GRSRecomLogger.FORMAT,
                            datefmt=GRSRecomLogger.DATE_FORMAT)
        return logging.getLogger(name)

    @classmethod
    def fileLogger(cls, name, logPath, extra=None):
        retLog = logging.getLogger(name)
        retLog.setLevel(logging.INFO)
        retLog.handlers = []
        h = logging.FileHandler(logPath, mode='a')
        formatter = logging.Formatter(GRSRecomLogger.FORMAT)
        h.setFormatter(formatter)
        retLog.addHandler(h)

        if extra:
            tf_log = logging.getLogger(extra)
            tf_log.setLevel(logging.INFO)
            tf_log.handlers = []
            tf_log.addHandler(h)
        return retLog


logger = GRSRecomLogger.stdLogger('aop_pytorch_kit')


def repalce_global_print():
    def my_print(*args, **kwargs):
        tmp_args = [str(arg) for arg in args]
        info = " ".join(tmp_args)
        logger.info(info)

    setattr(builtins, 'print', my_print)


def remove_handler_for_torch_init_process_group():
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
