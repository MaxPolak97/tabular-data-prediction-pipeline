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

WANDB_PROJECT = 'titanic_survived' # os.environ['WANDB_PROJECT'] # 'titanic_survived' 
WANDB_DATASET = 'titanic-dataset:latest' #os.environ['WANDB_DATASET'] # 'titanic-dataset:latest'

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
      y_train_preds_binary = np.where(y_train_preds >= 0.5, 1, 0)

      y_val_preds = pipeline.predict_proba(X_val)[:,1]
      y_val_preds_binary = np.where(y_val_preds >= 0.5, 1, 0)

      # Log Training metrics
      run.summary["train_acc"] = metrics.accuracy_score(y_train, y_train_preds_binary)

      # Log Validation metrics
      run.summary["val_acc"] = metrics.accuracy_score(y_val, y_val_preds_binary)
      wandb.log({"val_confusion_matrix" :wandb.sklearn.plot_confusion_matrix(y_val, 
                                                                   y_val_preds_binary, ['Dead','Survived'])})

      
      # Save model pipeline
      with open('model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

      date = datetime.now().strftime("%Y_%m_%d")
      model_artifact = wandb.Artifact(f'retrained_model_{date}', type='model',)
      model_artifact.add_file('model.pkl', )
      run.log_artifact(model_artifact, aliases=["latest", "candidate"])
      
if __name__ == "__main__":
   train()
    

