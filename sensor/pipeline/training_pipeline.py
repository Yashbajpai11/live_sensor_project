from sensor.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig
from sensor.exception import SensorException
from sensor.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact , ModelEvaluationArtifact, ModelPusherArtifact
from sensor.logger import logging 
import os,sys
from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidation
from sensor.components.data_transformation import DataTransformation
from sensor.components.model_trainer import ModelTrainer
from sensor.components.model_evaluation import ModelEvaluation
from sensor.components.model_pusher import ModelPusher

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
        
        
    def start_data_transformation(self,data_validation_artifact:DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config
            )
            data_transformation_artifact =  data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except  Exception as e:
            raise  SensorException(e,sys)
        
        
    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact):
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except  Exception as e:
            raise  SensorException(e,sys)
        
        
    def start_model_evaluation(self,data_validation_artifact:DataValidationArtifact,
                                 model_trainer_artifact:ModelTrainerArtifact,
                                ):
        try:
            model_eval_config = ModelEvaluationConfig(self.training_pipeline_config)
            model_eval = ModelEvaluation(model_eval_config, data_validation_artifact, model_trainer_artifact)
            model_eval_artifact = model_eval.initiate_model_evaluation()
            return model_eval_artifact
        except  Exception as e:
            raise  SensorException(e,sys)
        
    def start_model_pusher(self,model_eval_artifact:ModelEvaluationArtifact):
        try:
            model_pusher_config = ModelPusherConfig(training_pipeline_config=self.training_pipeline_config)
            model_pusher = ModelPusher(model_pusher_config, model_eval_artifact)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except  Exception as e:
            raise  SensorException(e,sys)
        

        
    def run_pipeline(self):
        try:

            TrainPipeline.is_pipeline_running=True


            data_ingestion_artifact:DataIngestionArtifact = self.start_data_ingestion()
            data_validation_artifact=self.start_data_validaton(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            model_eval_artifact = self.start_model_evaluation(data_validation_artifact, model_trainer_artifact)
            if not model_eval_artifact.is_model_accepted:
               raise Exception("Trained model is not better than the best model")
            model_pusher_artifact = self.start_model_pusher(model_eval_artifact)
            TrainPipeline.is_pipeline_running=False
            #self.sync_artifact_dir_to_s3()
            #self.sync_saved_model_dir_to_s3()
        except  Exception as e:
            #self.sync_artifact_dir_to_s3()
            TrainPipeline.is_pipeline_running=False
            raise  SensorException(e,sys)