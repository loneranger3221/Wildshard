from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_training():
    train_path, test_path = DataIngestion().initiate_data_ingestion()

    train_arr, test_arr, _, groups_train = (
        DataTransformation().initiate_data_transformation(train_path, test_path)
    )

    model_report, timing_report = ModelTrainer().initiate_model_trainer(
        train_arr,
        test_arr,
        groups_train,
    )

    print("Test metrics:")
    for model_name, metrics in model_report.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    print("\nTraining and inference times:")
    for model_name, timings in timing_report.items():
        print(f"{model_name}: {timings}")


if __name__ == "__main__":
    run_training()