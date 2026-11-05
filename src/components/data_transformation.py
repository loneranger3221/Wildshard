import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )
    label_encoder_file_path: str = os.path.join(
        "artifacts", "label_encoder.pkl"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                "AG-X", "AG-Y", "AG-Z",
                "Acc-X", "Acc-Y", "Acc-Z",
                "Gravity-X", "Gravity-Y", "Gravity-Z",
                "RR-X", "RR-Y", "RR-Z",
                "RV-X", "RV-Y", "RV-Z",
                "cos",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            return ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                ]
            )
        except Exception as e:
            logging.exception("Failed to create the data transformer")
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            groups_train = train_df.pop("block_id").to_numpy()
            test_df = test_df.drop(columns=["block_id"])
            
            target_column_name = "activity"
            X_train = train_df.drop(columns=[target_column_name])
            y_train = train_df[target_column_name]
            X_test = test_df.drop(columns=[target_column_name])
            y_test = test_df[target_column_name]

            preprocessor = self.get_data_transformer_object()
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            label_encoder = LabelEncoder()
            y_train_arr = label_encoder.fit_transform(y_train)
            y_test_arr = label_encoder.transform(y_test)

            # Keep the existing trainer interface: features followed by target.
            train_arr = np.c_[X_train_arr, y_train_arr]
            test_arr = np.c_[X_test_arr, y_test_arr]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor,
            )
            save_object(
                file_path=self.data_transformation_config.label_encoder_file_path,
                obj=label_encoder,
            )

            logging.info("Data transformation completed")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                groups_train,
            )
        except Exception as e:
            logging.exception("Failed to transform the data")
            raise CustomException(e, sys) from e