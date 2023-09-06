import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation


## Initilizing the Data Ingestion COnfig

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion has been started")

        try:
            df = pd.read_csv(os.path.join("notebooks/Data","gemstone.csv"))
            logging.info("Dataset read as pandas Dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info("Train Test Split")

            train_set,test_set = train_test_split(df,test_size=0.30,random_state=42)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data has been completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )






        except Exception as e:
            logging.info("Error occurred in Data ingestion stage")
            raise CustomException(e,sys)
            

'''
if __name__=='__main__':
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    '''