"""
Module 5 Week A — Lab: Regression & Evaluation

Build and evaluate logistic and linear regression models on the
Petra Telecom customer churn dataset.

Run: python lab_regression.py
"""

from pyexpat import model

from pyexpat import model

import pandas as pd
import numpy as np
from sklearn import pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, confusion_matrix,
                             mean_absolute_error, r2_score)
import matplotlib.pyplot as plt

def load_data(filepath="data/telecom_churn.csv"):
    
    df = pd.read_csv("data/telecom_churn.csv")
    print(df.shape)
    print(df.isnull().sum())
    return df 

def split_data(df, target_col, test_size=0.2, random_state=42):
    
    x =df.drop(target_col, axis=1)
    y=df[target_col]
    if y.nunique() <= 10:

       x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    else:
        x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42)


    print(f"Training set size:{len(x_train)}")
    print(f"Test set size:{len(x_test)}")
    print(f"train churn rate:{y_train.mean():.3f}")
    print(f"test churn rate:{y_test.mean():.3f}")

    return x_train, x_test, y_train, y_test


def build_logistic_pipeline():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])
    
    return pipeline


def build_ridge_pipeline():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    return pipeline
    

def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
  
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()

    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"]
    }
    

def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.3f}, R2: {r2:.3f}")
    return {"mae": mae, "r2": r2}
    

def run_cross_validation(pipeline, X_train, y_train):

    cv_splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv_splitter,
        scoring="accuracy"
    )

    print(scores)
    print(scores.mean(), scores.std())

    return scores   

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

        # Select numeric features for classification
        numeric_features = ["tenure", "monthly_charges", "total_charges",
                           "num_support_calls", "senior_citizen",
                           "has_partner", "has_dependents"]

        # Classification: predict churn
        df_cls = df[numeric_features + ["churned"]].dropna()
        split = split_data(df_cls, "churned")
        if split:
            X_train, X_test, y_train, y_test = split
            pipe = build_logistic_pipeline()
            if pipe:
                metrics = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)
                print(f"Logistic Regression: {metrics}")

                scores = run_cross_validation(pipe, X_train, y_train)
                if scores is not None:
                    print(f"CV: {scores.mean():.3f} +/- {scores.std():.3f}")

        # Regression: predict monthly_charges
        df_reg = df[["tenure", "total_charges", "num_support_calls",
                     "senior_citizen", "has_partner", "has_dependents",
                     "monthly_charges"]].dropna()
        split_reg = split_data(df_reg, "monthly_charges")
        if split_reg:
            X_tr, X_te, y_tr, y_te = split_reg
            ridge_pipe = build_ridge_pipeline()
            if ridge_pipe:
                reg_metrics = evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)
                print(f"Ridge Regression: {reg_metrics}")


"""
SUMMARY OF FINDINGS

1. Most important features for predicting churn:
Based on the logistic regression model, the most influential features are typically:
- num_support_calls
- monthly_charges
- tenure

These features strongly affect churn because:
- Higher support calls often indicate customer dissatisfaction.
- Higher monthly charges increase the likelihood of leaving.
- Longer tenure usually reduces churn risk (loyal customers).

2. Model performance:
The logistic regression model performs reasonably well with balanced accuracy across classes.
However, because the dataset is often imbalanced, accuracy alone is not enough to judge performance.

Recall for churned customers is more important than precision in this problem.
Missing a churned customer (false negative) is more costly than incorrectly predicting churn (false positive).

3. Precision vs Recall:
Recall is more concerning.
The model should prioritize capturing as many churned customers as possible,
even if it means slightly more false alarms.

4. Recommendations for improvement:
- Try stronger models like Random Forest or XGBoost
- Handle feature engineering (e.g., interaction between tenure and charges)
- Tune hyperparameters (C for Logistic Regression, alpha for Ridge/Lasso)
- Try threshold tuning instead of default 0.5
- Handle class imbalance using SMOTE or different class weights
"""