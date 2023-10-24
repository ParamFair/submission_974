from folktables import ACSDataSource, ACSPublicCoverage, ACSIncome
import pandas as pd

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

    return want_data

def load_income_data(states: list=['AL', 'AZ', 'FL',
                                    'GA', 'LA', 'MS',
                                    'NM', 'SC', 'TX', 
                                    'CA']):
    data_source = ACSDataSource(survey_year='2018',
                            horizon='1-Year',
                            survey='person')
                            
    data_all = data_source.get_data(states=states,
                                        download=True)

    features_cov, _, _ = ACSIncome.df_to_pandas(data_all)
    all_variables = set(list(features_cov.columns) + ['PINCP'])
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

    # Filter according to doc
    want_data = want_data.loc[want_data.AGEP >= 18,: ]
    want_data = want_data.loc[want_data.WKHP >= 1,: ]
    want_data = want_data.loc[want_data.PINCP >= 100,: ]

    return want_data