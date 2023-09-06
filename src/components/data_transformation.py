import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from src.utils import save_object

## Creating Dataclasss
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    #Initializing DataClass 
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    # creating function for Data Transformation 
    def get_data_transformation_obj(self):
        try:

            logging.info('Data Transformation Initiated') 

            ## Define which columns should be ordinal encoded and which should be scaled
            cat_columns = ['cut','color','clarity']
            num_columns = ['carat','depth','table','x','y','z']

            ## Define what should be the custom ranking  order of categories in cat columns.
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Pipeline initiated')

            ## Creating pipelines
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scalar',StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scalar',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,num_columns),
                ('cat_pipelihne',cat_pipeline,cat_columns)
            ])

            return preprocessor

            logging.info('Pipeline Completed')
        

        except Exception as e:
            logging.info('Error occured in Data Transforming')
            raise CustomException(e,sys)

    # Creating function to initialize the Transformation Process 
    def initiate_data_transformation(self, train_path,test_path):
        try:
            train_df = pd.read_csv('artifacts/train.csv')
            test_df = pd.read_csv('artifacts/test.csv')
             
            logging.info('reading train and test data is completed.')
            logging.info(f'Train DataFrame Head: \n {train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head: \n {test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            ## Preparing the Preprocessing Pipeline object
            preprocessing_obj = self.get_data_transformation_obj()

            #Define what should be Target and feature columns.
            target_columns_name=['price']
            drop_columns = ['Unnamed: 0', 'id']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_columns_name]


            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_columns_name]

            ## Applying the preprocessing pipeline on Train and test data 
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            ## Saving the preprocessor.pkl object
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj

            )

            logging.info("preprocessor pickle file saved.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path   
            )




        except Exception as e:
            logging.info('Error occured in Initiate data transformation')
            raise CustomException(e,sys)

