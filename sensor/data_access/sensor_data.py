import sys
from typing import Optional

import numpy as np
import pandas as pd
import json
from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.constant.training_pipeline import DATABASE_NAME
from sensor.exception import SensorException

class SensorData:
    """
    this class helpd to export entire mongo db record as a pandas dataframe
    """

    def __init__(self):
        """
        """
        try:
            self.mongo_client = MongoDBClient(database_name = DATABASE_NAME)

        except Exception as e:
            raise SensorException(e,sys)
        
    
    def save_csv_file(self,file_path,collection_name:str,database_name:Optional[str]=None):                      #firstly save data to mangodb 
        try:
            data_frame = pd.read_csv(file_path)
            data_frame.reset_index(drop=True,inplace=True)
            records = list(json.loads(data_frame.T.to_json()).values())
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]
            collection.insert_many(records)
            return len(records)
        except Exception as e:
            raise SensorException(e,sys)
        
    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        try:
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]
            
            print(f"Connected to collection: {collection.full_name}")
            documents = list(collection.find())
            print(f"Number of documents fetched: {len(documents)}")
    
            df = pd.DataFrame(documents)
    
            if "_id" in df.columns:
                df.drop("_id", axis=1, inplace=True)
    
            print("First 5 rows:")
            print(df.head())
    
            return df
        except Exception as e:
            print("MongoDB se data nikalte waqt error aaya:", e)
            raise SensorException(e, sys)
