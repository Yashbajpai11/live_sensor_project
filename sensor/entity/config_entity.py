from datetime import datetime                                                   # Abhi ka date aur time lene ke liye
import os                                                                        # Computer ke folders aur file paths ko handle karne ke liye
from sensor.constant  import training_pipeline                                    # Ek constant file se training pipeline ke fixed naam aur paths ko import kiya

class TrainingPipelineConfig:

    def __init__(self, timestamp=datetime.now()):                                         # Jab bhi class ka object banega, to current time le lenge
      timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")                                 # Time ko ek readable format mein convert kiya (month_day_year_hour_minute_second)

      self.pipeline_name: str = training_pipeline.PIPELINE_NAME                                 # Pipeline ka naam constants file se liya
      self.artifact_dir: str = os.path.join(training_pipeline.ARTIFACT_DIR,timestamp)                  # Ek folder banega jisme sari cheeze store hongi (har run ke liye alag folder time ke naam se)
      self.timestamp: str = timestamp                                                                     # Time ko object mein store bhi kiya future use ke liye

class DataIngestionConfig:                                                                                                                   # 👇 Ye class data ingestion ke liye sab folder aur file ka path batati hai
        def __init__(self,training_pipeline_config:TrainingPipelineConfig):                                                                # training_pipeline_config object se base artifact folder ka path mil raha hai
            self.data_ingestion_dir: str = os.path.join(
                training_pipeline_config.artifact_dir, training_pipeline.DATA_INGESTION_DIR_NAME                                            # Ye path banata hai: artifact/<timestamp>/data_ingestion
            )
            self.feature_store_file_path: str = os.path.join(
                self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR, training_pipeline.FILE_NAME                              # Ye file path banata hai: jahan MongoDB se aaya raw data store hoga
            )
            self.training_file_path: str = os.path.join(
                self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TRAIN_FILE_NAME                                  # Train data ka file path banata hai
            )
            self.testing_file_path: str = os.path.join(
                self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TEST_FILE_NAME                                      # Test data ka file path banata hai
            )
            self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION                                   # Ye value decide karti hai ki data ko kitne hisso me baatna hai (e.g. 80-20)
            self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME                                                              # MongoDB me jis collection se data uthana hai, uska naam
   

class DataValidationConfig:                                                                                                                  #Tum ek data-validation naam ka folder bana rahe ho, jiske andar kuch subfolders aur files automatically create honge.

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        #  1. Main data validation folder path (root folder)
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_VALIDATION_DIR_NAME
        )

        # 2. Folder path to store valid data
        self.valid_data_dir: str = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DIR
        )

        # 3. Folder path to store invalid data
        self.invalid_data_dir: str = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR
        )

        # 4. Valid training file path (CSV)
        self.valid_train_file_path: str = os.path.join(
            self.valid_data_dir,training_pipeline.TRAIN_FILE_NAME
        )

        # 5. Valid testing file path (CSV)
        self.valid_test_file_path: str = os.path.join(
            self.valid_data_dir, training_pipeline.TEST_FILE_NAME
        )

        # 6. Invalid training file path (CSV)
        self.invalid_train_file_path: str = os.path.join(
            self.invalid_data_dir,training_pipeline.TRAIN_FILE_NAME
        )

        # 7. Invalid testing file path (CSV)
        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_dir, training_pipeline.TEST_FILE_NAME
        )

        # 8. Path for storing data drift report (JSON or YAML file)
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )


class DataTransformationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join( training_pipeline_config.artifact_dir,training_pipeline.DATA_TRANSFORMATION_DIR_NAME )

        self.transformed_train_file_path: str = os.path.join( self.data_transformation_dir,training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TRAIN_FILE_NAME.replace("csv", "npy"),)
        
        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir,  training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TEST_FILE_NAME.replace("csv", "npy"), )
        
        self.transformed_object_file_path: str = os.path.join( self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipeline.PREPROCSSING_OBJECT_FILE_NAME,)
        

class ModelTrainerConfig:


    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR, 
            training_pipeline.MODEL_FILE_NAME
        )
        self.expected_accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold = training_pipeline.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD




