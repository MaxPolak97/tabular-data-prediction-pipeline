import wandb
import os
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn import metrics
from scipy.stats import ks_2samp

from wandb.xgboost import WandbCallback
import xgboost as xgb
import pickle

WANDB_PROJECT = os.environ['WANDB_PROJECT'] # 'titanic_survived' 
WANDB_DATASET = os.environ['WANDB_DATASET'] # 'titanic-dataset:latest'

target = 'Survived'

params = {'max_depth': 6 
  , 'min_child_weight': 1  
  , 'subsample': 1 
  , 'colsample_bytree': 1 
  , 'n_estimators': 100 
  , 'learning_rate': 0.3 
  , 'early_stopping_rounds': None 
  , 'random_state': 42
  , 'tree_method': 'hist'
  , 'device': 'cuda' 
    }


def train():
  with wandb.init(project=WANDB_PROJECT, job_type='retrain-model') as run:
      
      # Create Dataset directory
      root_dir = Path(sys.path[0])
      data_dir = root_dir / 'dataset'
      data_dir.mkdir(exist_ok=True)

      # Download latest dataset version
      data_art = run.use_artifact(WANDB_DATASET)
      data_dir = data_art.download(data_dir)
      
      # Read and split data into training and validation 
      df_data = pd.read_csv(f'{data_dir}/train.csv')

      df_train, df_val = train_test_split(df_data, 
                                        test_size=0.2,
                                        random_state=42, 
                                        stratify=df_data[[target]])
      

      # Feature selection to drop irrelevant features
      drop_list = [target, 'PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'] 

      X_train = df_train.drop(columns=drop_list)
      y_train = df_train[target]

      y_val = df_val[target]
      X_val = df_val.drop(columns=drop_list)

      # Create preprocessor
      preprocessor = ColumnTransformer(
          transformers=[("one_hot", OneHotEncoder(), selector(dtype_include="object"))], # One Hot Encoding
          remainder="passthrough",  # Leave numerical features unchanged
      )

      # Create XGBoost classifier
      model = xgb.XGBClassifier(**params,
                                  callbacks=[WandbCallback(log_model=False, log_feature_importance=False)])

      # Create pipeline
      pipeline = Pipeline([("preprocessor", preprocessor), # Preprocessing
                          ("classifier", model) # 
                          ])
    
      # Fit the pipeline on the training data
      pipeline.fit(X_train, y_train)

      # Get train and validation predictions
      y_train_preds = pipeline.predict_proba(X_train)[:,1]
      y_val_preds = pipeline.predict_proba(X_val)[:,1]

      # Log Validation metrics
      fpr, tpr, thresholds = metrics.roc_curve(y_val, y_val_preds)
      optimal_idx = np.argmax(tpr - fpr)
      optimal_threshold = thresholds[optimal_idx]

      y_val_preds_binary = np.where(y_val_preds >= optimal_threshold, 1, 0)

      run.summary["val_acc_0.5"] = metrics.accuracy_score(y_val, np.where(y_val_preds >= 0.5, 1, 0))
      run.summary[f"val_acc_best_trh"] = metrics.accuracy_score(y_val, y_val_preds_binary)
      run.summary[f"val_precision"] = metrics.precision_score(y_val, y_val_preds_binary)
      run.summary[f"val_recall"] = metrics.recall_score(y_val, y_val_preds_binary)
      run.summary[f"val_f1score"] = metrics.f1_score(y_val, y_val_preds_binary)
      run.summary["val_auc"] = metrics.roc_auc_score(y_val, y_val_preds)
      
      run.summary["val_matthews_corrcoef"] = metrics.matthews_corrcoef(y_val, y_val_preds_binary)
      run.summary["val_cohen_kappa_score"] = metrics.cohen_kappa_score(y_val, y_val_preds_binary)
      run.summary["val_brier_loss"] = metrics.brier_score_loss(y_val, y_val_preds)
      run.summary["val_log_loss"] = -(y_val * np.log(y_val_preds)
                                      + (1-y_val) * np.log(1-y_val_preds)).sum() / len(y_val)
      
      ks_stat, ks_pval = ks_2samp(y_val_preds[y_val==1], y_val_preds[y_val==0])
      run.summary["val_ks_2samp"] = ks_stat
      run.summary["val_ks_pval"] = ks_pval

      run.log({"confusion_matrix" :wandb.sklearn.plot_confusion_matrix(y_val, 
                                                                   y_val_preds_binary, ['Dead','Survived'])})
      
      # Save model pipeline

      # Get the current date and time
      date = datetime.now().strftime("%Y_%m_%d")

      with open('model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

      model_artifact = wandb.Artifact(f'retrained_model_{date}', type='model')
      model_artifact.add_file('model.pkl')
      run.log_artifact(model_artifact)

if __name__ == "__main__":
   train()
    

