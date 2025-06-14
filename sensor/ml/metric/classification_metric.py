from sensor.entity.artifact_entity import ClassificationMetricArtifact                                       # Custom class jo f1, precision, recall score ko ek saath store karta hai
from sensor.exception import SensorException                                                                # Custom error handler jo exception ke sath system ka info bhi deta hai (debugging ke liye)
from sklearn.metrics import f1_score, precision_score, recall_score                                                   # Scikit-learn se evaluation metrics import kiye gaye hain
import os, sys                                                                                                                                    # OS aur system-level functionality ke liye modules import

def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:                                                         # Function to calculate and return classification metrics
    try:                                                                                                                                  # Try block — agar koi error aaye to handle kiya ja sake
        model_f1_score = f1_score(y_true, y_pred)                                                                                           # Actual aur predicted labels se F1 Score calculate karta hai
        model_recall_score = recall_score(y_true, y_pred)                                                                                     # Recall Score calculate karta hai
        model_precision_score = precision_score(y_true, y_pred)                                                                                   # Precision Score calculate karta hai

        classsification_metric = ClassificationMetricArtifact(                                                                                             # ClassificationMetricArtifact object create kiya jisme 3 scores jaayenge
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score
        )
        return classsification_metric                                                                                                                                      # Final result return kar diya as a single object
    except Exception as e:                                                                                                                                  # Agar upar koi bhi error aaye to yahan handle hoga
        raise SensorException(e, sys)                                                                                                                       # Custom exception raise karega with error message and system info
