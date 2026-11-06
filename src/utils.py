import pickle 
import sys
import os
from src.exception import CustomException
import time 
import json

import dill
'''Dill is a python package which is used to serialize and deserialize python objects. It is an extension of the pickle module and it can serialize a wider range of python objects than pickle.'''

'''Main packages'''
import numpy as np
import pandas as pd

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
 

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV,StratifiedGroupKFold
    
def evaluate_models(X_train, y_train, X_test, y_test, groups_train, models, params):
    try:
        model_report = {}
        fitted_models = {}
        timing_report = {}
        cv_scores = {}
        
        checkpoint_path = os.path.join("artifacts", "model_results.json")
        temp_path = checkpoint_path + ".tmp"

        def save_checkpoint(status):
            os.makedirs("artifacts", exist_ok=True)
            checkpoint = {
                "status": status,
                "completed_models": list(model_report.keys()),
                "model_report": model_report,
                "timing_report": timing_report,
                "cv_scores": cv_scores,
            }
            with open(temp_path, "w", encoding="utf-8") as file:
                json.dump(checkpoint, file, indent=2)
            os.replace(temp_path, checkpoint_path)

        save_checkpoint("running")

        cv = StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=42,
        )
        labels = np.unique(np.concatenate((y_train, y_test)))

        for name, model in models.items():
            grid = GridSearchCV(
                estimator=model,
                param_grid=params[name],
                scoring="f1_macro",
                cv=cv,
                verbose=1,
            )

            start = time.perf_counter()
            grid.fit(X_train, y_train, groups=groups_train)
            train_time = time.perf_counter() - start

            best_model = grid.best_estimator_
            fitted_models[name] = best_model
            cv_scores[name] = float(grid.best_score_)

            start = time.perf_counter()
            y_test_pred = best_model.predict(X_test)
            inference_time = time.perf_counter() - start

            model_report[name] = {
                "accuracy": float(accuracy_score(y_test, y_test_pred)),
                "precision_macro": float(precision_score(
                    y_test, y_test_pred, average="macro", zero_division=0
                )),
                "recall_macro": float(recall_score(
                    y_test, y_test_pred, average="macro", zero_division=0
                )),
                "f1_macro": float(f1_score(
                    y_test, y_test_pred, average="macro", zero_division=0
                )),
                "confusion_matrix": confusion_matrix(
                    y_test, y_test_pred, labels=labels
                ).tolist(),
            }
            timing_report[name] = {
                "train_time_sec": train_time,
                "inference_time_sec": inference_time,
            }
            save_checkpoint("running")
            
        save_checkpoint("complete")
        
        return model_report, fitted_models, timing_report, cv_scores
    except Exception as e:
        raise CustomException(e, sys)
    
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)
        