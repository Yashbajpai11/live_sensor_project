from sensor.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig
from sensor.exception import SensorException
from sensor.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from sensor.logger import logging 
import os,sys
from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidation

class TrainPipeline:
    is_pipeline_running=False                                                                                                                 #yeh batata hai ki pipeline abhi chal rahi hai ya nahi (True ya False)
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()                                                                             # jisme timestamp ke saath pipeline ka naam aur folder ka path hota hai

        
    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)                                      # Yeh config banata hai jisme path, file names, split ratio sab hota hai
            logging.info("Starting data ingestion")                                                                                                            #Yeh log batata hai ki ab data ingestion start ho gaya

            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)                                                                           # Ab actual data ingestion component banate hain
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()                                                               #Yeh MongoDB se data laata hai, csv me save karta hai, aur split karta hai
            logging.info(f"Data ingestion completed and artifact: {data_ingestion_artifact}")                                                         #Yeh batata hai ki kaunsa file kaha save hua (train.csv, test.csv)

            return data_ingestion_artifact
        except  Exception as e:
            raise  SensorException(e,sys)
        
    def start_data_validaton(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config = data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        except  Exception as e:
            raise  SensorException(e,sys)
        

        
    def run_pipeline(self):
        try:
            data_ingestion_artifact:DataIngestionArtifact = self.start_data_ingestion()
            data_validation_artifact=self.start_data_validaton(data_ingestion_artifact=data_ingestion_artifact)
            #data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            #model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            #model_eval_artifact = self.start_model_evaluation(data_validation_artifact, model_trainer_artifact)
            #if not model_eval_artifact.is_model_accepted:
            #    raise Exception("Trained model is not better than the best model")
            #model_pusher_artifact = self.start_model_pusher(model_eval_artifact)
            #TrainPipeline.is_pipeline_running=False
            #self.sync_artifact_dir_to_s3()
            #self.sync_saved_model_dir_to_s3()
        except  Exception as e:
            #self.sync_artifact_dir_to_s3()
            #TrainPipeline.is_pipeline_running=False
            raise  SensorException(e,sys)