from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV 
from dataclasses import dataclass
from src.exception import Sys_error
from src.components.data_ingestion import timestamp
from src.utils import load_numpy_array_data, save_obj, write_yaml_file, read_yaml_file
from src.logger import logging
import pandas as pd
import sys, os

models = {
    "Linear_Regression": LinearRegression(),
    "K-Neighbors_Regressor": KNeighborsRegressor(),
    "Decision_Tree": DecisionTreeRegressor(),
    "Random_Forest_Regressor": RandomForestRegressor(),
    "AdaBoost_Regressor": AdaBoostRegressor(),
    "Gradient_Boosting_Regressor": GradientBoostingRegressor(),
    "Xgboost_Regressor": XGBRegressor(),
    "CatBoosting_Regressor": CatBoostRegressor(verbose=False)}

params = {
    "Linear_Regression": {"fit_intercept":[True]},
    "K-Neighbors_Regressor": {"n_neighbors":[3,4,5]},
    "Decision_Tree": {"criterion":["squared_error", "friedman_mse", "absolute_error", "poisson"], "max_depth":[3,4,5]},
    "Random_Forest_Regressor": {'n_estimators': [8,16,32,64,128,256]},
    "AdaBoost_Regressor": {'learning_rate':[.1,.01,0.5,.001],
                            # 'loss':['linear','square','exponential'],
                            'n_estimators': [8,16,32,64,128,256]},
    "Gradient_Boosting_Regressor": {# 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                            'learning_rate':[.1,.01,.05,.001],
                            'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                            # 'criterion':['squared_error', 'friedman_mse'],
                            # 'max_features':['auto','sqrt','log2'],
                            'n_estimators': [8,16,32,64,128,256]},
    "Xgboost_Regressor": {'learning_rate':[.1,.01,.05,.001],
                            'n_estimators': [8,16,32,64,128,256]},
    "CatBoosting_Regressor": {'depth': [6,8,10],
                            'learning_rate': [0.01, 0.05, 0.1],
                            'iterations': [30, 50, 100]}}


@dataclass
class ModelTrainerConfig:
    try:
        saved_model_path:str = os.path.join("artifacts", timestamp, "model_trainer", "model.pkl")
        performance_report_path = os.path.join("artifacts", timestamp, "model_trainer", "report.yaml")
    except Exception as e:
        print(Sys_error(e, sys))

class ModelTrainer:
    def __init__(self):
        try:
            self.model_trainer_config = ModelTrainerConfig()
        except Exception as e:
            print(Sys_error(e, sys))
    
    def training_model(self, X_train, y_train, X_test, model, params):
        try:
            gs = GridSearchCV(model, params, cv = 3)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            return train_pred, test_pred
        except Exception as e:
            print(Sys_error(e, sys))

    def evaluate_model(self, X, y):
        try:
            score = r2_score(X, y)
            return score
        except Exception as e:
            print(Sys_error(e, sys))

    def save_model(self, report_path, X_train, y_train):
        try:
            report = read_yaml_file(report_path)
            df = pd.DataFrame(report)
            df['Train Score'] = df['Train Score'].astype("float32")
            df['Test Score'] = df['Test Score'].astype("float32")
            df_sorted = df.sort_values(by=['Train Score'], ascending=False)
            if df_sorted.iloc[0,1]<0.6:
                return False
            for i in range(len(df_sorted)):
                if abs(df_sorted.iloc[i,1]-df_sorted.iloc[i,2])<0.015:
                    best_model = df_sorted.iloc[i,0]
                    logging.info("Best model found: {}".format(best_model))
                    break
            model = models[best_model]
            model.fit(X_train, y_train)
            save_obj(self.model_trainer_config.saved_model_path, model)
            return True
        except Exception as e:
            print(Sys_error(e, sys))

    def initialise_model_trainer(self, train_arr_path, test_arr_path):
        try:
            model_report = {}
            model_lst, train, test = [], [], []
            logging.info("Loading the Transformed Data")
            train_arr = load_numpy_array_data(train_arr_path)
            test_arr = load_numpy_array_data(test_arr_path)
            
            logging.info("Splitting the data")
            X_train, y_train, X_test ,y_test = (train_arr[:,:-1], train_arr[:,-1], test_arr[:,:-1], test_arr[:,-1])
            
            logging.info("Training the model")
            for i in range(len(models.keys())):
                logging.info(f"Training Model {[*models.keys()][i]}")
                model = models[[*models.keys()][i]]
                para = params[[*params.keys()][i]]
                train_pred, test_pred = self.training_model(X_train, y_train, X_test, model, para)
                train_score = self.evaluate_model(y_train, train_pred)
                test_score = self.evaluate_model(y_test, test_pred)
                model_lst.append([*models.keys()][i])
                train.append(str(train_score))
                test.append(str(test_score))
            model_report["Model"] = model_lst
            model_report["Train Score"] = train
            model_report["Test Score"] = test
            
            logging.info("Saving the Model Performance Report")
            write_yaml_file(self.model_trainer_config.performance_report_path, model_report)
            logging.info("Getting the Best Model")
            out = self.save_model(self.model_trainer_config.performance_report_path, X_train, y_train)
            if not out:
                logging.info("No Best Model Found")
                return (self.model_trainer_config.performance_report_path)
            else:
                logging.info("Model Trainer completed")
                return (self.model_trainer_config.performance_report_path, self.model_trainer_config.saved_model_path)
        except Exception as e:
            print(Sys_error(e, sys))