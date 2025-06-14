from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.config_entity import DataIngestionConfig
from sensor.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import os, sys

from sensor.data_access.sensor_data import SensorData
from sensor.utils.main_utils import read_yaml_file
from sensor.constant.training_pipeline import SCHEMA_FILE_PATH


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise SensorException(e, sys)

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Fetch data from MongoDB and save it as a CSV in the feature store.
        """
        try:
            logging.info("Exporting data from MongoDB to feature store.")
            sensor_data = SensorData()
            dataframe = sensor_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )

            if dataframe.empty:
                raise SensorException("Exported dataframe is empty!", sys)

            # Save the DataFrame to CSV
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logging.info(f"Data exported to feature store at {feature_store_file_path}")

            return dataframe
        except Exception as e:
            raise SensorException(e, sys)

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Split the DataFrame into train and test sets and save them.
        """
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )

            logging.info("Performed train-test split on the dataframe.")

            # Ensure the training directory exists
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Save train and test sets
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )

            logging.info(f"Train and test files saved at {dir_path}")
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Execute the full data ingestion pipeline: fetch from DB → clean → split → save.
        """
        try:
            # Step 1: Load data from MongoDB
            dataframe = self.export_data_into_feature_store()

            print("\n\n=== Dataframe Sample ===\n", dataframe.head(), "\n\n")


            # Step 2: Clean column names
            dataframe.columns = dataframe.columns.astype(str).str.strip()

            # Step 3: Drop columns specified in schema
            drop_columns = [col.strip() for col in self._schema_config.get("drop_columns", [])]
            existing_drop_columns = [col for col in drop_columns if col in dataframe.columns]
            dataframe = dataframe.drop(existing_drop_columns, axis=1)
            logging.info(f"Dropped columns: {existing_drop_columns}")

            # Step 4: Split into train and test
            self.split_data_as_train_test(dataframe=dataframe)

            # Step 5: Return artifact
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            logging.info(f"Data ingestion artifact created: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise SensorException(e, sys)
