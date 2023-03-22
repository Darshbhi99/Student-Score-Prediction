from src.pipeline.training_pipeline import TrainingPipeline
from src.exception import Sys_error
from src.utils import load_obj
import pandas as pd
import numpy as np
import sys, os

def preprocessing_data(data, path):
    try:
        df = pd.DataFrame(data, index=[0])
        preprocessor = load_obj(path)
        arr = preprocessor.transform(df)
        return arr
    except Exception as e:
        print(Sys_error(e, sys))


def predict_marks(data:dict, new_model = False):
    try:
        latest_file = os.listdir("artifacts")[np.argmax([int(i) for i in os.listdir("artifacts")])]
        pre_path = os.path.join("artifacts", latest_file, "data_transformation", "preprocessor.pkl")
        if new_model == True:
            training_pipeline = TrainingPipeline()
            path = training_pipeline.initialize_training_pipeline()
            model = load_obj(path[1])
            pre_data = preprocessing_data(data, pre_path)
            output = model.predict(pre_data)
            return output
        else:
            mod_path = os.path.join("artifacts", latest_file, "model_trainer", "model.pkl")
            model = load_obj(mod_path)
            pre_data = preprocessing_data(data, pre_path)
            output = model.predict(pre_data)
            return output
    except Exception as e:
        print(Sys_error(e, sys))