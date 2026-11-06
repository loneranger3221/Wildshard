import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass 
from src.utils import save_object, evaluate_models

'''Main packages for model training '''

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array, groups_train):
        try:
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1].astype(int)
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1].astype(int)

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "KNN Classifier": KNeighborsClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
                "Random Forest Classifier": RandomForestClassifier(random_state=42),
                "AdaBoost Classifier": AdaBoostClassifier(random_state=42),
                "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
                "XGBoost Classifier": XGBClassifier(
                    eval_metric="mlogloss", random_state=42
                ),
                "MLP Classifier": MLPClassifier(max_iter=500, random_state=42),}

            params = {
                "Logistic Regression": {"C": [0.1, 1.0]},
                "KNN Classifier": {"n_neighbors": [5, 9]},
                "Decision Tree Classifier": {"criterion": ["gini", "entropy"]},
                "Random Forest Classifier": {"max_depth": [20]},
                "AdaBoost Classifier": {},
                "Gradient Boosting Classifier": {
                    "learning_rate": [0.1],
                    "n_estimators": [50],
                },
                "XGBoost Classifier": {
                    "learning_rate": [0.1],
                    "n_estimators": [50],
                    "max_depth": [6],
                },
                "MLP Classifier": {
                    "hidden_layer_sizes": [(64,)],
                    "alpha": [0.0001],
                },
            }

            model_report, fitted_models, timing_report, cv_scores = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                groups_train=groups_train,
                models=models,
                params=params,
            )

            # Choose using grouped cross-validation, not the held-out test scores.
            best_model_name = max(cv_scores, key=cv_scores.get)
            best_model = fitted_models[best_model_name]

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            logging.info(
                "Best model by grouped CV macro-F1: %s (%.4f)",
                best_model_name,
                cv_scores[best_model_name],
            )
            return model_report, timing_report

        except Exception as e:
            logging.exception("Exception occurred during model training")
            raise CustomException(e, sys) from e
