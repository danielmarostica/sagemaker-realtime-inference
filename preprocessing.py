import argparse
import joblib
import os
import json

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

from io import StringIO

from sagemaker_containers.beta.framework import (worker, encoders)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    args, _ = parser.parse_known_args()

    input_data_path = os.path.join(args.train, 'titanic.csv')

    print('Reading input data from {}'.format(input_data_path))
    
    df = pd.read_csv(input_data_path)
    df.drop('Survived', axis=1, inplace=True, errors='ignore') # predict_fn will reinsert this column

    preprocessor = make_column_transformer(
        (StandardScaler(), ['Age', 'Fare', 'Parch', 'SibSp', 'Family_Size']),
        (OneHotEncoder(sparse=False), ['Embarked', 'Pclass', 'Sex']))
    
    preprocessor.fit(df)
    
    joblib.dump(preprocessor, os.path.join(args.model_dir, 'preprocessor.joblib'))

def model_fn(model_dir):
    '''
    The name of this function has to be 'model_fn' and receive a 'model_dir' parameter, so it can be recognized by SageMaker for training and inference.
    '''
    
    preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.joblib'))
    return preprocessor   

def input_fn(input_data, content_type):
    '''
    Parse input data payload.
    '''
    if content_type == 'text/csv':
        df = pd.read_csv(StringIO(input_data))
        return df
    
    elif content_type == 'application/json':
        df = pd.read_json(input_data)
        return df
    
    else:
        raise ValueError('{} content type is not supported by this script!'.format(content_type))
        
def predict_fn(input_data, model):
    '''
    Preprocess input data

    We implement this because the default predict_fn uses .predict(), but this model is a preprocessor
    so we want to use transform().

    '''
    features = model.transform(input_data)

    # Return the label (as the first column) and the set of features.
    try:
        return np.insert(features, 0, input_data['Survived'], axis=1)  
    except:
        return features    