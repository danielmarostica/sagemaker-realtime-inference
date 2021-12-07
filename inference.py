#!/usr/bin/env python  
import argparse
import joblib
import os
import json

from io import StringIO

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sagemaker_containers.beta.framework import (worker, encoders)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n-estimators', type=int, default=10)
    parser.add_argument('--min-samples-leaf', type=int, default=3)

    # variáveis de ambiente do sagemaker, que podem ser substituídas por parâmetros ao executar localmente
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--train_file', type=str, default='titanic.csv.out')

    args, _ = parser.parse_known_args()

    train_df = pd.read_csv(os.path.join(args.train, args.train_file))

    X_train = train_df.iloc[:, 1:]
    y_train = train_df.iloc[:, 0]

    model = RandomForestClassifier(
        n_estimators=args.n_estimators, min_samples_leaf=args.min_samples_leaf, n_jobs=-1)

    model.fit(X_train, y_train)

    path = os.path.join(args.model_dir, 'model.joblib')
    joblib.dump(model, path)

def model_fn(model_dir):
    '''
    The name of this function has to be 'model_fn' and receive a 'model_dir' parameter, so it can be recognized by SageMaker for training and inference.
    '''
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return model

def output_fn(prediction, accept):
    '''Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as 'accept' so the next
    container can read the response payload correctly.
    '''
    if accept == 'application/json':
        json_output = {'Survived': prediction.tolist()}
        
        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    
    else:
        raise RuntimeException('{} accept type is not supported by this script.'.format(accept))



