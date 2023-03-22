import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import Sys_error
from src.logger import logging
from dataclasses import dataclass
from datetime import datetime

timestamp = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
@dataclass
class DataIngestionConfig:
    """ This class will contain all configuration for data ingestion"""
    try:
        train_path = os.path.join("artifacts", timestamp, "data_ingestion", "train.csv")
        test_path = os.path.join("artifacts", timestamp, "data_ingestion", "test.csv")
        raw_data_path = os.path.join("artifacts", timestamp, "data_ingestion", "data.csv")
    except Exception as e:
        print(Sys_error(e, sys))

class DataIngestion:
    """ This class will contain all the methods for data ingestion """
    def __init__(self):
        try:
            self.data_ingestion_config = DataIngestionConfig()
        except Exception as e:
            print(Sys_error(e, sys))

    def initiaize_data_ingestion(self):
        logging.info("Entering the Data Ingestion Pipeline")
        try:
            df = pd.read_csv(os.path.join("data", "student_data.csv"))
            logging.info("Saving the Raw data")
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header= True)
            logging.info("Splitting the data")
            train_set, test_set = train_test_split(df, test_size= 0.2, shuffle=True, random_state= 42)
            logging.info("Saving the train and test_data")
            train_set.to_csv(self.data_ingestion_config.train_path, index=False, header= True)
            test_set.to_csv(self.data_ingestion_config.test_path, index=False, header= True)
            logging.info("Data Ingestion Completed")
            return (self.data_ingestion_config.train_path, self.data_ingestion_config.test_path)
        except Exception as e:
            print(Sys_error(e, sys))