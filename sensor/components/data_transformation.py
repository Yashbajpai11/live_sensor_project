# Required imports
import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek  # Imbalanced data balance karta hai
from sklearn.impute import SimpleImputer  # Missing values ko fill karega
from sklearn.preprocessing import RobustScaler  # Outliers ko handle karega
from sklearn.pipeline import Pipeline  # Data cleaning steps ko ek sequence mein run karega

# Custom imports
from sensor.constant.training_pipeline import TARGET_COLUMN
from sensor.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from sensor.entity.config_entity import DataTransformationConfig
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.ml.model.estimator import TargetValueMapping
from sensor.utils.main_utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, 
                 data_transformation_config: DataTransformationConfig):
        """
        Ye class 2 cheezon ko input le rahi:
        - data_validation_artifact: jisme clean train/test file paths honge
        - data_transformation_config: jisme transformed files save karne ke paths honge
        """
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise SensorException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        # CSV file read karta hai aur DataFrame return karta hai
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SensorException(e, sys)
        
        
    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        # Yeh ek pipeline return karta hai: Imputer + RobustScaler
        try:
            robust_scaler = RobustScaler()  # Outliers ko normalize karta hai
            simple_imputer = SimpleImputer(strategy="constant", fill_value=0)  # NaN ko 0 bana deta hai

            preprocessor = Pipeline(
                steps=[
                    ("Imputer", simple_imputer),
                    ("RobustScaler", robust_scaler)
                ]
            )
            return preprocessor
        except Exception as e:
            raise SensorException(e, sys) from e

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        'na' strings ko NaN mein convert karta hai aur numeric columns ko proper dtypes mein convert karta hai
        """
        try:
            # 'na' strings ko actual NaN values mein convert karo
            df = df.replace('na', np.nan)
            
            # All columns (except target) ko numeric mein convert karo
            for col in df.columns:
                if col != TARGET_COLUMN:  # Target column ko skip karo
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logging.info(f"Data cleaned: converted 'na' strings to NaN and numeric conversion done")
            return df
        except Exception as e:
            raise SensorException(e, sys) from e
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:

        try:
            # Step 1: Cleaned data read karo
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # Step 1.5: 'na' strings ko handle karo
            train_df = self.clean_data(train_df)
            test_df = self.clean_data(test_df)

            # Step 2: Features aur target alag karo
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN].replace(TargetValueMapping().to_dict())

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN].replace(TargetValueMapping().to_dict())

            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)

            # Step 3: Data ko transform karo
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            smt = SMOTETomek(sampling_strategy="minority")

            # Step 4: Imbalanced data ko balance karo
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                transformed_input_test_feature, target_feature_test_df
            )

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path, array=train_arr
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path, array=test_arr
            )
            save_object(
                self.data_transformation_config.transformed_object_file_path, preprocessor_object
            )

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys) from e