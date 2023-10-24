import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def load_compas(location_string='./data/compas/compas-scores-two-years.csv'):
    # Data can be recovered from https://github.com/propublica/compas-analysis
    df_complete = pd.read_csv('data/compas/compas-scores-two-years.csv')

    # analogous to the original article, remove some observations
    df_want = df_complete.copy()
    df_want = df_want.loc[(df_want.days_b_screening_arrest <= 30), :]
    df_want = df_want.loc[(df_want.days_b_screening_arrest >= -30), :]
    df_want = df_want.loc[df_want.is_recid != -1, :]
    df_want = df_want.loc[df_want.c_charge_degree != 'O', :]
    df_want = df_want.loc[~df_want.score_text.isna(), :]

    # Run dummies 
    # Create dummies (makes it easier for lightgbm)
    df_want['possession_dummy'] = df_want.c_charge_desc.apply(lambda x: 'poss' in str(x).lower())*1
    df_want['theft_dummy'] = df_want.c_charge_desc.apply(lambda x: 'theft' in str(x).lower())*1
    df_want['driv_dummy'] = df_want.c_charge_desc.apply(lambda x: 'driv' in str(x).lower())*1
    df_want['battery_dummy'] = df_want.c_charge_desc.apply(lambda x: 'batt' in str(x).lower())*1
    df_want['assault_dummy'] = df_want.c_charge_desc.apply(lambda x: 'assault' in str(x).lower())*1
    df_want['weapon_dummy'] = df_want.c_charge_desc.apply(lambda x: 'weap' in str(x).lower())*1
    df_want['firearm_dummy'] = df_want.c_charge_desc.apply(lambda x: 'arm' in str(x).lower())*1

    df_want.drop(columns='c_charge_desc', inplace=True)

    df_want['male_dummy'] = np.where(df_want.sex == 'Male', 1, 0)
    df_want.drop(columns='sex', inplace=True)

    return df_want


def prepare_compas(target='is_recid', 
                   drop_list = None, 
                   seed=42):
    
    data_complete = load_compas()

    features = data_complete.drop(columns=['is_recid', 'is_violent_recid'])
    labels = data_complete.loc[:,[target]]

    # drop all columns with too many missings
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
                    ('numerical', scaler_, numeric_cols)
                    ], 
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


    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy(), transformer



