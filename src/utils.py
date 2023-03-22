from src.exception import Sys_error
import numpy as np
import yaml
import pickle as pkl
import sys, os


def save_obj(file_path, obj):
    """Saving the object"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pkl.dump(obj, f)
    except Exception as e:
        print(Sys_error(e, sys))

def load_obj(file_path):
    """Loading the object"""
    try:
        if not os.path.exists(file_path):
            print("This file does not exist")
        obj = pkl.load(open(file_path, 'rb'))
        return obj
    except Exception as e:
        print(Sys_error(e, sys))


def save_numpy_array_data(file_path:str, array: np.array):
    """Saving numpy array"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            np.save(f, array)
    except Exception as e:
        print(Sys_error(e, sys))

def load_numpy_array_data(file_path:str):
    """Loading numpy array"""
    try:
        with open(file_path, 'rb') as f:
            return np.load(f)
    except Exception as e:
        print(Sys_error(e, sys))

def write_yaml_file(file_path:str, data)-> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            yaml.dump(data, f)
    except Exception as e:
        print(Sys_error(e, sys))

def read_yaml_file(file_path:str)-> dict:
    try:
        if not os.path.exists(file_path):
            print("This file does not exist")
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(Sys_error(e, sys))