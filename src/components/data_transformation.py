import os, sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.utils import save_numpy_array_data, save_obj
from src.components.data_ingestion import timestamp
from src.exception import Sys_error
from src.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

@dataclass
class DataTransformationConfig:
    """This class contain configuration for data transformation"""
    try:
        train_arr_filepath = os.path.join("artifacts", timestamp, "data_transformation", "train.npy") 
        test_arr_filepath = os.path.join("artifacts", timestamp, "data_transformation", "test.npy")
        preprocessor_obj_file_path = os.path.join("artifacts", timestamp, "data_transformation", "preprocessor.pkl")
    except Exception as e:
        print(Sys_error(e, sys))

class DataTransformation:
    """This class contain methods for data transformation"""
    def __init__(self):
        try:
            self.data_transformation_config = DataTransformationConfig()
        except Exception as e:
            print(Sys_error(e, sys))
    
    def get_data_transformer_object(self)->object:
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = ["gender", "parental_level_of_education", 
                                    "race_ethnicity", "lunch", "test_preparation_course"]
            
            logging.info(f"Numerical Columns are {numerical_columns} and Categorical columns are {categorical_columns}")
            num_pipeline = Pipeline(steps=[("impute", SimpleImputer(strategy='median')),
                                                ("scaler", StandardScaler(with_mean= False))])
            
            cat_pipeline = Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),
                                                ('encoder', OneHotEncoder()),
                                                ('scaler', StandardScaler(with_mean = False))])
            
            preprocessor = ColumnTransformer([("num_pipeline", num_pipeline, numerical_columns),
                                                ("cat_pipeline", cat_pipeline, categorical_columns)])
            return preprocessor
        except Exception as e:
            print(Sys_error(e, sys))

    def initialise_data_transformation(self, train_path, test_path):
        try:
            logging.info("Initialising data transformation")
            
            train = pd.read_csv(train_path)
            input_feature_train_df = train.drop("math_score" ,axis = 1)
            target_feature_train_df = train["math_score"]
            
            test = pd.read_csv(test_path)
            input_feature_test_df = test.drop("math_score" ,axis = 1)
            target_feature_test_df = test["math_score"]
            
            logging.info("Creating preprocessor object and transforming data")
            preprocessor =  self.get_data_transformer_object()
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving the Train array, test array and Preprocessor Model")
            save_numpy_array_data(self.data_transformation_config.train_arr_filepath, train_arr)
            save_numpy_array_data(self.data_transformation_config.test_arr_filepath, test_arr)
            save_obj(obj = preprocessor, file_path = self.data_transformation_config.preprocessor_obj_file_path)
            logging.info("Data Transformation Completed")
            
            return (self.data_transformation_config.train_arr_filepath, 
                    self.data_transformation_config.test_arr_filepath,
                    self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            print(Sys_error(e, sys))