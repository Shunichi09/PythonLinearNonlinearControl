import argparse
import datetime
import json
import os
import sys
import six
import pickle
from logging import DEBUG, basicConfig, getLogger, FileHandler, StreamHandler, Formatter, Logger


def make_logger(save_dir):
    """
    Args:
        save_dir (str): save directory
    """
    # base config setting
    basicConfig(
        format='[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # mypackage log level
    logger = getLogger("PythonLinearNonlinearControl")
    logger.setLevel(DEBUG)

    # file handler
    log_path = os.path.join(save_dir, "log.txt")
    file_handler = FileHandler(log_path)
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(Formatter('%(message)s'))
    logger.addHandler(file_handler)

    # sh handler
    # sh_handler = StreamHandler()
    # logger.addHandler(sh_handler)


def int_tuple(s):
    """ transform str to tuple
    Args:
        s (str): strings that you want to change
    Returns:
        tuple
    """
    return tuple(int(i) for i in s.split(','))


def bool_flag(s):
    """ transform str to bool flg
    Args:
        s (str): strings that you want to change
    """
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)


def file_exists(path):
    """ Check file existence on given path
    Args:
        path (str):  path of the file to check existence
    Returns:
        file_existence (bool): True if file exists otherwise False        
    """
    return os.path.exists(path)


def create_dir_if_not_exist(outdir):
    """ Check directory existence and creates new directory if not exist
    Args:  
        outdir (str): path of the file to create directory
    RuntimeError:
        file exists in outdir but it is not a directory
    """
    if file_exists(outdir):
        if not os.path.isdir(outdir):
            raise RuntimeError('{} is not a directory'.format(outdir))
        else:
            return
    os.makedirs(outdir)


def write_text_to_file(file_path, data):
    """ Write given text data to file
    Args:
        file_path (str): path of the file to write data
        data (str): text to write to the file
    """
    with open(file_path, 'w') as f:
        f.write(data)


def read_text_from_file(file_path):
    """ Read given file as text
    Args:
        file_path (str): path of the file to read data
    Returns
        data (str): text read from the file
    """
    with open(file_path, 'r') as f:
        return f.read()


def save_pickle(file_path, data):
    """ pickle given data to file
    Args:
        file_path (str): path of the file to pickle data
        data (): data to pickle
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file_path):
    """ load pickled data from file
    Args:
        file_path (str): path of the file to load pickled data
    Returns:
        data (): data pickled in file
    """
    with open(file_path, 'rb') as f:
        if six.PY2:
            return pickle.load(f)
        else:
            return pickle.load(f, encoding='bytes')


def prepare_output_dir(base_dir, args, time_format='%Y-%m-%d-%H%M%S'):
    """ prepare a directory with current datetime as name.
    created directory contains the command and args when the script was called as text file.
    Args:
        base_dir (str): path of the directory to save data
        args (dict): arguments when the python script was called
        time_format (str): datetime format string for naming directory to save data
    Returns:
        out_dir (str): directory to save data
    """
    time_str = datetime.datetime.now().strftime(time_format)
    outdir = os.path.join(base_dir, time_str)

    create_dir_if_not_exist(outdir)

    # Save all the arguments
    args_file_path = os.path.join(outdir, 'args.txt')
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    write_text_to_file(args_file_path, json.dumps(args))

    # Save the command
    argv_file_path = os.path.join(outdir, 'command.txt')
    argv = ' '.join(sys.argv)
    write_text_to_file(argv_file_path, argv)

    return outdir
