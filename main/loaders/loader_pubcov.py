import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from folktables import ACSDataSource, ACSPublicCoverage

def load_coverage_data(states: list=['AL', 'AZ', 'FL',
                                    'GA', 'LA', 'MS',
                                    'NM', 'SC', 'TX', 
                                    'CA']):
    data_source = ACSDataSource(survey_year='2018',
                            horizon='1-Year',
                            survey='person')
                            
    data_all = data_source.get_data(states=states,
                                        download=True)

    features_cov, _, _ = ACSPublicCoverage.df_to_pandas(data_all)
    all_variables = set(list(features_cov.columns) + ['PUBCOV'])
    want_data = data_all.loc[:, list(all_variables)]

    # Racast race variables
    repl_dict = { 1: 'white',
                  2: 'black',
                  3: 'native_american',
                  4: 'native_alaskan',
                  5: 'native_both',
                  6: 'asian',
                  7: 'pacific_islander',
                  8: 'other',
                  9: 'mixed'}

    want_data.RAC1P = want_data.RAC1P.map(repl_dict)

    # Filter according to doc (note we filter at 60k not at 30k as in the 
    # standard application, as we introduce a sensitive variable later on)
    want_data = want_data.loc[want_data.PINCP <= 60000,: ]
    want_data = want_data.loc[want_data.AGEP < 65,: ]
    want_data = want_data.loc[want_data.PINCP > 0,: ]

    return want_data


def prepare_pubcov(data_complete, 
                   target='PUBCOV',
                   seed=42, 
                   drop_list = []):
    
    # split into features and 
    # labels
    features = data_complete.drop(columns=[target])
    labels = data_complete.loc[:,[target]]

    # drop columns as described in experimental setup
    missing_cols = list((data_complete
                         .isna()
                         .sum(axis=0)[data_complete.isna()
                                      .sum(axis=0) > 1]
                                      .index))
    
    drop_cols = missing_cols + drop_list
    features = features.loc[:, [col for col in features.columns if not col in drop_cols]]

    # Run basic preprocessing
    non_numeric_cols = list(features.select_dtypes('object').columns)
    numeric_cols = list(features.select_dtypes('float').columns)

    onehot_ = OneHotEncoder()
    scaler_ = StandardScaler()

    transformer = ColumnTransformer(
        transformers=[('categoricals', onehot_, non_numeric_cols), 
                    ('numerical', scaler_, numeric_cols)], 
                    remainder='passthrough', 
                    sparse_threshold=0)
    
    X_train, X_test, y_train, y_test = train_test_split(features, 
                                                        labels, 
                                                        test_size=0.2, 
                                                        random_state=seed)
    
    # Transform features 
    transformer.fit(X_train)

    X_train = transformer.transform(X_train)
    X_test = transformer.transform(X_test)

    # Return transformer to isolate idx of given datasets
    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy(), transformer


