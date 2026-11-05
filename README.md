# Research Project Report
## Comparative Study of ML/DL Models for Smartphone-Based Human Activity Recognition (Wild-SHARD)

---

## 1. Dataset

**Title:** Wild-SHARD: Smartphone Sensor-based Human Activity Recognition Dataset in Wild
**Source:** IEEE DataPort, DOI: 10.21227/dc2h-a428
**Authors:** Nurul Amin Choudhury, Badal Soni (National Institute of Technology Silchar)
**Released:** June 3, 2024
**File used:** `wildshard.xlsx`

**What it contains:** A Human Activity Recognition (HAR) dataset collected in an uncontrolled, real-world ("in the wild") setting — as opposed to the lab-controlled conditions of most HAR benchmarks (e.g., UCI-HAR). Sensor modules (accelerometer + gyroscope) were mounted in the front pockets of 40 adult subjects of varying age, gender, weight, and height, across multiple different smartphone models (Samsung Galaxy F62, Galaxy A30s, Poco X2, OnePlus 9 Pro, and others), sampled at 100 Hz. Subjects performed six Activities of Daily Living (ADLs) naturally: **Sitting, Walking, Standing, Running, Upstairs, Downstairs.**

---

## 2. Confirmed Dataset Schema (from the uploaded file)

**Sheet:** `wildshard` (single sheet). **Rows:** 483,895. **Columns:** 17 (16 numeric sensor features + 1 activity label). **Missing values:** none.

| Column | Description |
|---|---|
| `AG-X`, `AG-Y`, `AG-Z` | Acceleration due to gravity (3-axis) |
| `Acc-X`, `Acc-Y`, `Acc-Z` | Linear acceleration (3-axis) |
| `Gravity-X`, `Gravity-Y`, `Gravity-Z` | Gravity vector (3-axis) |
| `RR-X`, `RR-Y`, `RR-Z` | Rotational rate / gyroscope (3-axis) |
| `RV-X`, `RV-Y`, `RV-Z` | Rotational vector (3-axis) |
| `cos` | Cosine of the rotational vector angle |
| `activity` | Target label — one of 6 classes |

**Class distribution (mildly imbalanced, ~1.24:1 max:min ratio — not severe enough to require SMOTE):**

| Activity | Count |
|---|---|
| Walking | 90,194 |
| Sitting | 86,574 |
| Running | 80,447 |
| Standing | 79,947 |
| Upstairs | 74,302 |
| Downstairs | 72,431 |

**Structural finding — important for methodology:** The rows are **not independently shuffled**. They form **178 contiguous same-activity blocks** (sequential sensor readings from individual recording sessions/trials). Adjacent rows within a block are near-duplicates of each other (same activity, consecutive 10ms timesteps). This matters directly for how the train/test split must be done — see Section 3.

---

## 3. Novelty Check — What's Already Published on This Dataset

Unlike the energy-flexibility dataset used earlier, **Wild-SHARD is not untouched.** A search turned up a directly relevant existing paper:

> **Nafea, A. A. et al., "A Comparative Study of Machine Learning Techniques for Human Activity Recognition Using Smartphone Sensor Data," Passer Journal of Basic and Applied Sciences, 7(2), 2025.**

This paper uses the *exact same* Wild-SHARD dataset (confirmed: same 483,895-instance, 16-feature structure) and compares **SVM, Decision Tree, Naive Bayes, KNN, and LDA**, with SMOTE class balancing and 5-fold cross-validation. Reported result: **KNN best at 96.86% accuracy**, DT second at 92.33%, NB and LDA performing poorly (64.18% and 41.92%).

Additionally, the dataset's own authors (Choudhury & Soni) have separately published CNN-LSTM-based deep learning models for HAR in uncontrolled environments (IEEE TII 2022, IEEE Sensors Journal 2023) using closely related data collection methodology, predating this specific IEEE DataPort release.

**Verification note:** the leakage claim above is a well-founded inference from Nafea et al.'s stated methodology (plain 5-fold CV, no mention of session/block grouping) combined with the data's confirmed block structure — not a confirmed fact about their literal code, which isn't published. Given 178 blocks averaging ~2,700 rows against folds of ~96,800 rows, an ungrouped fold split can't realistically avoid cutting through session boundaries either way, so the risk holds regardless of the exact implementation.

**Conclusion: a plain "compare classical ML models on Wild-SHARD" study would be redundant** — it's already been done, on this exact dataset, fairly recently. Two genuine gaps remain, both low-effort to address:

1. **Model coverage gap** — Nafea et al. tested 5 classical linear/instance/tree/probabilistic models but included **no ensemble-boosting methods (Random Forest, Gradient Boosting, XGBoost) and no neural network** of any kind.
2. **Methodology gap (more substantive)** — The dataset's 178-block contiguous structure (Section 2) creates a leakage risk that a plain, ungrouped train/test split does not account for. **What's directly confirmed from Nafea et al.'s paper:** their Methods section states they used standard 5-fold cross-validation (K-Fold, k=5) after SMOTE balancing, and makes no mention anywhere of session/subject/block-aware grouping or of the data's sequential structure. **What's not confirmed:** their paper doesn't link any code, so the exact implementation (e.g., whether folds were shuffled) can't be verified directly. **Why the risk holds regardless:** with 178 blocks averaging ~2,700 rows each and folds of ~96,800 rows, a fold that size cannot align with block boundaries by chance — shuffled or not, an ungrouped K-Fold over this data would almost certainly place temporally-adjacent readings from the same recording session on both sides of the train/test boundary. This is a well-documented pitfall in sensor-based HAR literature. A **block-aware / group-based split** (splitting whole recording sessions between train and test, never splitting a session across both) avoids this and gives a more realistic estimate of real-world generalization.

---

## 4. Research Idea (finalized)

**Framing:** A block-leakage-aware comparative study of ML and neural models for smartphone-based HAR on Wild-SHARD, extending the existing published comparison by (a) benchmarking model families it did not test — ensemble bagging/boosting and a neural network — and (b) using a methodologically sounder train/test split that avoids the temporal-leakage risk inherent to the dataset's block structure.

**Research question:** *How do ensemble and neural-network classifiers compare against the previously benchmarked classical models on Wild-SHARD, and does accounting for the dataset's session-block structure (rather than a naive random split) change the picture?*

This keeps essentially the same low-effort template structure as before — swap regressors for classifiers, same `GridSearchCV` training loop — while giving you a defensible answer to "why is this not just repeating an existing paper."

---

## 5. What We Will Actually Do

1. Load the `wildshard` sheet; confirm structure (already done — Section 2).
2. Detect the 178 contiguous activity blocks and assign each row a `block_id`.
3. Split into train/test using `GroupShuffleSplit` grouped by `block_id` (80/20), so no session is split across both sets.
4. Preprocess: scale all 16 numeric sensor features; encode the `activity` target.
5. Train and compare **8 classification models** spanning classical ML, boosting, and a neural network.
6. Evaluate on accuracy, macro-precision, macro-recall, macro-F1, and confusion matrices, plus training/inference time.
7. Compare results against Nafea et al.'s reported numbers as a discussion point — expect the block-aware split to show a (likely lower, more realistic) accuracy gap versus their ungrouped 5-fold CV numbers, which is itself a finding worth discussing in the paper.

---

## 6. Models to Train and Compare (8 total)

| # | Model | Family | Tested by Nafea et al.? |
|---|---|---|---|
| 1 | Logistic Regression | Linear baseline | No (they used LDA, not logistic regression) |
| 2 | KNN Classifier | Instance-based | **Yes** — kept for direct comparison |
| 3 | Decision Tree Classifier | Tree-based | **Yes** — kept for direct comparison |
| 4 | Random Forest Classifier | Ensemble — bagging | **No — gap filled** |
| 5 | AdaBoost Classifier | Ensemble — boosting | **No — gap filled** |
| 6 | Gradient Boosting Classifier | Ensemble — boosting | **No — gap filled** |
| 7 | **XGBoost Classifier** | Ensemble — boosting (added) | **No — gap filled** |
| 8 | **MLP Classifier** | Neural network (added) | **No — gap filled** |

KNN and Decision Tree are deliberately kept (rather than dropped) so your results table can directly show whether their previously-reported near-97%/92% accuracy holds up once the block-aware split removes the suspected leakage risk — that comparison is itself a talking point for your results section.

---

## 7. Step-by-Step Pipeline

1. **Data acquisition & structural check** — *(done)* Confirmed 483,895 rows, 16 features, 6 classes, no missing values, 178 contiguous session blocks.
2. **Problem formulation** — *(done)* Target = `activity` (6-class classification); features = all 16 sensor columns.
3. **Block detection** — In pandas: `block_id = (df['activity'] != df['activity'].shift()).cumsum()`. This is the one genuinely new step versus the energy-dataset template, and it's a single line.
4. **Preprocessing pipeline** — `StandardScaler` on all 16 numeric features (no categorical branch needed — every feature here is a continuous sensor reading); `LabelEncoder` on the target.
5. **Train/test split** — `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` grouped by `block_id`, instead of the template's plain `train_test_split`. This is the key methodological upgrade over the existing paper.
6. **Model training** — Run all 8 models through the same `evaluate_models()` / `GridSearchCV` loop pattern from the template, adapted to classifiers.
7. **Metrics & timing** — Accuracy, macro-precision, macro-recall, macro-F1, confusion matrix per model, plus training/inference time (via `time.time()`).
8. **Comparative discussion** — Compare your KNN/Decision Tree numbers directly against Nafea et al.'s reported 96.86%/92.33% to discuss the likely effect of the block-aware split (framed as a hypothesis their paper's methodology doesn't rule out, not a confirmed fact about their code); discuss where ensemble/neural models land relative to the classical baseline.
9. **Results write-up** — Comparison table across all 8 models, confusion matrices for the top 2–3 models (activities like Upstairs/Downstairs are often confused with each other and with Walking — worth calling out), bar charts of metrics per model, and the leakage-methodology discussion as a distinguishing contribution.

---

## 8. How the Template Maps to This Project — With Real Code

Same skeleton as the energy-forecasting adaptation — ingestion → transformation → training → evaluation — but since this is **classification, not regression**, the model classes and metrics change more than they did for the energy dataset. The `GridSearchCV`/pipeline *pattern* itself is identical.

### `src/components/data_ingestion.py`
```python
import os
import sys
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method')
        try:
            df = pd.read_excel(
                'notebooks/data/wildshard.xlsx',
                sheet_name='wildshard'
            )

            # Detect contiguous same-activity recording blocks (178 in the full dataset)
            df['block_id'] = (df['activity'] != df['activity'].shift()).cumsum()

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            # Block-aware split: no recording session is split across train and test
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(gss.split(df, groups=df['block_id']))
            train_set = df.iloc[train_idx].drop(columns=['block_id'])
            test_set = df.iloc[test_idx].drop(columns=['block_id'])

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info('Ingestion of the data is completed')
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e, sys)
```

### `src/components/data_transformation.py` — numeric-only pipeline, label-encoded target
```python
numerical_columns = [
    'AG-X', 'AG-Y', 'AG-Z', 'Acc-X', 'Acc-Y', 'Acc-Z',
    'Gravity-X', 'Gravity-Y', 'Gravity-Z',
    'RR-X', 'RR-Y', 'RR-Z', 'RV-X', 'RV-Y', 'RV-Z', 'cos'
]
# No categorical_columns branch needed — every feature is a continuous sensor reading.
target_column_name = 'activity'

# In initiate_data_transformation(), after separating input/target:
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
target_feature_train_df = label_encoder.fit_transform(target_feature_train_df)
target_feature_test_df = label_encoder.transform(target_feature_test_df)
# save_object the label_encoder too, alongside the preprocessor, so predictions can be decoded back to activity names
```
Everything else in this file (the numeric `Pipeline` — impute + `StandardScaler` — and `save_object` call) is unchanged from the template; only the categorical branch of the `ColumnTransformer` is dropped since there are no categorical features here.

### `src/components/model_trainer.py` — classifiers instead of regressors, +2 added
```python
import time
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN Classifier': KNeighborsClassifier(),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'Random Forest Classifier': RandomForestClassifier(),
    'AdaBoost Classifier': AdaBoostClassifier(),
    'Gradient Boosting Classifier': GradientBoostingClassifier(),
    'XGBoost Classifier': XGBClassifier(),                        # added
    'MLP Classifier': MLPClassifier(max_iter=500),                 # added
}

params = {
    'Decision Tree Classifier': {'criterion': ['gini', 'entropy']},
    'Random Forest Classifier': {'n_estimators': [50, 100, 200]},
    'Gradient Boosting Classifier': {
        'learning_rate': [.1, .05, .01],
        'n_estimators': [50, 100, 200]
    },
    'XGBoost Classifier': {                                        # added
        'learning_rate': [.1, .05],
        'n_estimators': [100, 200]
    },
    'MLP Classifier': {                                            # added
        'hidden_layer_sizes': [(64,), (64, 32)],
        'alpha': [0.0001, 0.001]
    },
    'Logistic Regression': {'C': [0.1, 1.0, 10.0]},
    'KNN Classifier': {'n_neighbors': [5, 7, 9, 11]},
    'AdaBoost Classifier': {},
}
```

### `src/utils.py` — classification metrics + timing in `evaluate_models`
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    model_report = {}
    fitted_models = {}
    timing_report = {}
    for name, model in models.items():
        param = params[name]
        grid = GridSearchCV(estimator=model, param_grid=param, cv=3, verbose=1)

        t0 = time.time()
        grid.fit(X_train, y_train)
        train_time = time.time() - t0

        best_model = grid.best_estimator_
        fitted_models[name] = best_model

        t0 = time.time()
        y_test_pred = best_model.predict(X_test)
        inference_time = time.time() - t0

        model_report[name] = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision_macro': precision_score(y_test, y_test_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_test_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_test_pred, average='macro'),
        }
        timing_report[name] = {'train_time_sec': train_time, 'inference_time_sec': inference_time}

    return model_report, fitted_models, timing_report
```

### Everything else — unchanged
`src/exception.py`, `src/logger.py`, `src/utils.py`'s `save_object`/`load_object`, and the overall `GridSearchCV` pattern stay exactly as in the original template.

### Dropped
- `application.py` (Flask app) — not needed for the paper.

### Filled in (was empty in the original template)
- `src/pipeline/train_pipeline.py` — orchestrates ingestion → transformation → training → evaluation as one script, same as recommended for the energy-dataset adaptation.

### Summary of all changes from the original template
- Data source: CSV → Excel (`wildshard` sheet)
- Task type: **regression → classification** (the single biggest structural change vs. the energy-dataset adaptation)
- Feature set: demographic/exam scores → 16 continuous IMU sensor readings (no categorical branch needed)
- Target: math score → `activity` (6-class label, label-encoded)
- Train/test split: ungrouped row-level split → **block-aware `GroupShuffleSplit`** (addresses a leakage risk not accounted for in prior published work on this dataset, based on what their methodology section states)
- Models: 6 regressors → 8 classifiers (Logistic Regression, KNN, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, MLP)
- Metrics: R² → Accuracy, macro-Precision, macro-Recall, macro-F1, confusion matrix + training/inference time
- `train_pipeline.py`: empty → actual orchestrated entrypoint
- Flask app: kept → dropped (optional demo only)
- Everything else (exception handling, logging, save/load pattern, `GridSearchCV` pattern): **unchanged**

---

## 9. Summary / Pitch for Your Guide

> We're using the IEEE DataPort "Wild-SHARD" dataset (released June 2024) — a smartphone accelerometer/gyroscope Human Activity Recognition dataset collected from 40 subjects in real-world, uncontrolled conditions across six activities (Sitting, Walking, Standing, Running, Upstairs, Downstairs), sampled at 100 Hz, totaling ~484,000 labeled readings.
>
> An existing comparative-ML paper (Nafea et al., 2025) has already benchmarked five classical classifiers (SVM, Decision Tree, Naive Bayes, KNN, LDA) on this exact dataset, reporting up to 96.86% accuracy, using standard 5-fold cross-validation. Our project extends this in two ways: first, by benchmarking model families that paper did not test — ensemble methods (Random Forest, Gradient Boosting, XGBoost) and a neural network (MLP) — alongside the previously tested KNN and Decision Tree for direct comparison. Second, we identified that the dataset's rows are organized into 178 contiguous recording sessions rather than independent shuffled samples, and that the existing paper's methodology section makes no mention of accounting for this structure. An ordinary, ungrouped split — whether random or the standard K-fold CV that paper describes — risks leaking near-duplicate, temporally adjacent readings between train and test given how much smaller each session block is than each fold, which would inflate reported accuracy. We can't verify their exact code since none is published, but we address this risk directly by using a session/block-aware split that keeps each recording session entirely within either train or test, giving a more realistic estimate of how these models would generalize to genuinely unseen recording sessions.
>
> We'll report accuracy, precision, recall, and F1 for all 8 models, along with training and inference time, and discuss how our block-aware results compare to the previously published cross-validation numbers — both as a model comparison and as a methodological point about evaluation practice in sensor-based HAR research.

---

*Compiled from project planning discussion and the uploaded dataset file `wildshard.xlsx` · Dataset DOI: 10.21227/dc2h-a428 · Related published comparison: Nafea et al., Passer J. Basic Appl. Sci. 7(2), 2025*
