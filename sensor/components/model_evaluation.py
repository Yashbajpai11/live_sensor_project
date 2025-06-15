from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.artifact_entity import DataValidationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact
from sensor.entity.config_entity import ModelEvaluationConfig
import os, sys
from sensor.ml.metric.classification_metric import get_classification_score
from sensor.ml.model.estimator import SensorModel
from sensor.utils.main_utils import save_object, load_object, write_yaml_file
from sensor.ml.model.estimator import ModelResolver
from sensor.constant.training_pipeline import TARGET_COLUMN
from sensor.ml.model.estimator import TargetValueMapping
import pandas as pd

class ModelEvaluation:

    # Jab class ka object banate hain, tab yeh constructor config aur artifacts ko set karta hai
    def __init__(self, model_eval_config: ModelEvaluationConfig,
                       data_validation_artifact: DataValidationArtifact,
                       model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise SensorException(e, sys)

    # Ye function check karega ki trained model purane model se better hai ya nahi
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            # Validated train aur test file ka path le rahe hain
            valid_train_file_path = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path

            # Train aur test data ko load karke ek hi dataframe mein combine kar diya
            train_df = pd.read_csv(valid_train_file_path)
            test_df = pd.read_csv(valid_test_file_path)
            df = pd.concat([train_df, test_df])

            # Target column ke string labels ko number mein convert kar diya (ex: pos → 1, neg → 0)
            y_true = df[TARGET_COLUMN]
            y_true.replace(TargetValueMapping().to_dict(), inplace=True)

            # Target column ko features se hata diya (kyunki prediction ke time pe target chahiye nahi hota)
            df.drop(TARGET_COLUMN, axis=1, inplace=True)

            # Naya train hua model ka path le kar us model ko load kar rahe hain
            train_model_file_path = self.model_trainer_artifact.trained_model_file_path
            train_model = load_object(file_path=train_model_file_path)

            # ModelResolver ka object banaya jo purana best model dhoondhne mein help karega
            model_resolver = ModelResolver()
            is_model_accepted = True

            # Agar pehle se koi best model hi nahi mila to current model ko hi accept kar lo
            if not model_resolver.is_model_exists():
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted,
                    improved_accuracy=None,
                    best_model_path=None,
                    trained_model_path=train_model_file_path,
                    train_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact,
                    best_model_metric_artifact=None
                )
                logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact

            # Agar purana model mila to uska path lekar usko bhi load kar rahe hain
            latest_model_path = model_resolver.get_best_model_path()
            latest_model = load_object(file_path=latest_model_path)

            # Dono models (naya aur purana) se same data pe prediction kar rahe hain
            y_trained_pred = train_model.predict(df)
            y_latest_pred = latest_model.predict(df)

            # Dono predictions ka performance score (F1-score) calculate kar rahe hain
            trained_metric = get_classification_score(y_true, y_trained_pred)
            latest_metric = get_classification_score(y_true, y_latest_pred)

            # Performance improvement check kar rahe hain (new_model - old_model)
            improved_accuracy = trained_metric.f1_score - latest_metric.f1_score

            # Agar improvement threshold se zyada hai to model accept karo, warna nahi
            if self.model_eval_config.change_threshold < improved_accuracy:
                is_model_accepted = True
            else:
                is_model_accepted = False

            # Final evaluation artifact banate hain jisme dono models ke scores aur paths hote hain
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_accuracy=improved_accuracy,
                best_model_path=latest_model_path,
                trained_model_path=train_model_file_path,
                train_model_metric_artifact=trained_metric,
                best_model_metric_artifact=latest_metric
            )

            # Evaluation artifact ko dictionary mein convert karke YAML file mein save kar rahe hain
            model_eval_report = model_evaluation_artifact.__dict__
            write_yaml_file(self.model_eval_config.report_file_path, model_eval_report)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise SensorException(e, sys)
