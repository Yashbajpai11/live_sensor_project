import shutil
from sensor.constant.training_pipeline import SCHEMA_FILE_PATH
from sensor.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from sensor.entity.config_entity import DataValidationConfig
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.utils.main_utils import read_yaml_file
from sensor.utils.main_utils import write_yaml_file   # ya jahan pe likha ho wahan se
from scipy.stats import ks_2samp
import pandas as pd
import os,sys


class DataValidation:

    # Constructor: isme ingestion aur config artifacts aate hain + schema.yaml ko read krta hai
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                        data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact  # Yeh train/test file ka path laata hai
            self.data_validation_config = data_validation_config    # Yeh drift report file path laata hai
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH) # schema.yaml file read kar ke config save kar raha
        except Exception as e:
            raise  SensorException(e,sys)

    # 🔁 (Optional) Function to drop columns jinki standard deviation 0 hai (currently empty)
    def drop_zero_std_columns(self,dataframe):
        pass

    # ✅ Yeh function check karta hai ki dataframe ke columns ka count schema ke equal hai ya nahi
    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns = len(self._schema_config["columns"])  # Expected column count
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data frame has columns: {len(dataframe.columns)}")
            
            # Agar match ho gaya toh return True, warna False
            if len(dataframe.columns)==number_of_columns:
                return True
            return False
        except Exception as e:
            raise SensorException(e,sys)

    # 🔢 Yeh check karta hai ki schema ke numerical columns dataframe me present hain ya nahi
    def is_numerical_column_exist(self,dataframe:pd.DataFrame)->bool:
        try:
            numerical_columns = self._schema_config["numerical_columns"]   # schema.yaml se expected numerical columns
            dataframe_columns = dataframe.columns                          # actual dataframe ke columns

            numerical_column_present = True
            missing_numerical_columns = []
            
            # Har numerical column ko check karte hain ki dataframe me hai ya nahi
            for num_column in numerical_columns:
                if num_column not in dataframe_columns:
                    numerical_column_present = False
                    missing_numerical_columns.append(num_column)
            
            # Missing columns ki info log me print hoti hai
            logging.info(f"Missing numerical columns: [{missing_numerical_columns}]")
            return numerical_column_present
        except Exception as e:
            raise SensorException(e,sys)

    # 📖 Yeh static function hai jo CSV file ko read karke dataframe return karta hai
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SensorException(e,sys)

    # 🔄 Yeh function train aur test data ke beech **data drift** check karta hai using KS test
    def detect_dataset_drift(self, base_df, current_df, threshold=0.05):
        try:
            drift_report = {}
            numerical_columns = base_df.select_dtypes(include=['int64', 'float64']).columns
            
            for column in numerical_columns:
                d1 = base_df[column]
                d2 = current_df[column]
    
                # Drop NaNs and convert all to float for ks_2samp
                d1_clean = pd.to_numeric(d1.dropna(), errors='coerce')
                d2_clean = pd.to_numeric(d2.dropna(), errors='coerce')
    
                d1_clean = d1_clean.dropna()
                d2_clean = d2_clean.dropna()
    
                if len(d1_clean) == 0 or len(d2_clean) == 0:
                    drift_report[column] = {
                        "p_value": None,
                        "drift_status": "Insufficient data"
                    }
                    continue
    
                is_same_dist = ks_2samp(d1_clean, d2_clean)
                drift_status = is_same_dist.pvalue > threshold
    
                drift_report[column] = {
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": drift_status
                }
    
            # line 104 in data_validation.py
            write_yaml_file(
                     file_path=self.data_validation_config.drift_report_file_path,
                     data=drift_report
                )

            return True
    
        except Exception as e:
            raise SensorException(e, sys)


    # 🚀 Ye main function hai jo pura data validation pipeline execute karta hai
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            error_message = ""  # Saare errors collect karke final me raise karenge

            # Ingested train/test files ka path le rahe hain
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Train aur Test data read karte hain
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            # Step 1️⃣: Check number of columns in train
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message = f"{error_message}Train dataframe does not contain all columns.\n"

            # Step 2️⃣: Check number of columns in test
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message = f"{error_message}Test dataframe does not contain all columns.\n"

            # Step 3️⃣: Check numerical columns in train
            status = self.is_numerical_column_exist(dataframe=train_dataframe)
            if not status:
                error_message = f"{error_message}Train dataframe does not contain all numerical columns.\n"

            # Step 4️⃣: Check numerical columns in test
            status = self.is_numerical_column_exist(dataframe=test_dataframe)
            if not status:
                error_message = f"{error_message}Test dataframe does not contain all numerical columns.\n"

            # Agar koi bhi error mila toh raise kar denge
            if len(error_message) > 0:
                raise Exception(error_message)

            # Step 5️⃣: Run data drift check
            status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)

            # Step 6️⃣: Sab kuch sahi gaya toh ek DataValidationArtifact return karenge
            data_validation_artifact = DataValidationArtifact(
                validation_status = status,
                valid_train_file_path = self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path = self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path = None,
                invalid_test_file_path = None,
                drift_report_file_path = self.data_validation_config.drift_report_file_path,
            )

            # Final output ko log kar dete hain
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise SensorException(e,sys)
