from src.exception import Sys_error
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import sys

class TrainingPipeline:
    def __init__(self, DataIngestion = DataIngestion(), DataTransformation = DataTransformation(), ModelTrainer = ModelTrainer()):
        try:
            self.data_ingestion = DataIngestion
            self.data_transformation = DataTransformation
            self.model_trainer = ModelTrainer
        except Exception as e:
            logging.info(e)
            print(Sys_error(e, sys))
    
    def initialize_training_pipeline(self):
        try:
            DI_path = self.data_ingestion.initiaize_data_ingestion()
            DT_path = self.data_transformation.initialise_data_transformation(DI_path[0], DI_path[1])
            MT_path = self.model_trainer.initialise_model_trainer(DT_path[0], DT_path[1])
            return MT_path
        except Exception as e:
            print(Sys_error(e, sys))