# author: Jenit Jain, Shaun Hutchinson, Ritisha Sharma
# date: 2022-11-23

"""
A script to clean and transform the raw input dataset and split it into train and test sets

Usage: preprocess.py --input_file_path=<input_file_path> --preprocessed_out_dir=<preprocessed_out_dir> --processed_out_dir=<processed_out_dir>

Options:
--input_file_path=<input_file_path>                 Takes the input file path for the original file (this is a required positional argument)
--preprocessed_out_dir=<preprocessed_out_dir>       Takes a directory path to dump the preprocessed data (this is a required positional argument)
--processed_out_dir=<processed_out_dir>             Takes a directory path to dump the processed and encoded data (this is a required positional argument)

"""

# Example:
# python src/preprocess.py --input_file_path=data/raw/drug_consumption.csv --preprocessed_out_dir=data/preprocessed --processed_out_dir=data/processed


import os
from docopt import docopt
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
    OneHotEncoder,
    PolynomialFeatures
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import json


def main(in_path, preprocessed_out_dir, processed_out_dir):
    '''
    Given the input dataset, it cleans it and saves a preprocessed version.
    It then applies column transformations and stores the transformed train and test sets
    
    Parameters
    ----------
    in_path : str
        File location of the raw dataset
    preprocessed_out_dir : str
        File directory to store the cleaned dataset
    processed_out_dir : str
        File directory to store the transformed dataset
    
    Examples
    --------
    >>> main(in_path, preprocessed_dir, processed_dir)
    '''
    #The following JSON file contains all the mappings required for data-cleaning 
    mappings = json.load(open("src/data_mapping.json", "r"))

    df = pd.read_table(in_path, index_col=0, names=mappings["column_headers"], delimiter=',')
    
    # We will drop overclaimers since, there answers might not truly be accurate
    df = df.drop(df[df['Semer'] != 'CL0'].index)
    
    for key, values in mappings["categories"].items():
        #The float values are keys and are henced saved as string values in the JSON object
        #Hence we are casting them back to float here
        values = {float(k):v for k,v in values.items()}
        df[key] = df.replace({key:values})[key]

    train_df, test_df = train_test_split(df, train_size=0.8, random_state=522)
    
    train_df.to_csv(os.path.join(preprocessed_out_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(preprocessed_out_dir, "test.csv"), index=False)
    
    #Perform col transformation
    preprocessor =  make_column_transformer(
                        (make_pipeline(PolynomialFeatures(degree=3),
                                       StandardScaler()), mappings['numerical']),
                        (OrdinalEncoder(categories = [
                            list(mappings['categories']['Age'].values()),
                            list(mappings['categories']['Education'].values()),
                            list(mappings['categories']['Impulsiveness'].values()),
                            list(mappings['categories']['SensationSeeking'].values()),
                        ]), mappings['ordinal']),
                        (OneHotEncoder(drop='if_binary', dtype=int, handle_unknown='ignore'), mappings['categorical']),
                        ("drop", mappings['drop'])
                    )
    preprocessor.fit(train_df)
    
    X_train_enc = pd.DataFrame(preprocessor.transform(train_df),
                               columns=preprocessor.get_feature_names_out())
    X_test_enc = pd.DataFrame(preprocessor.transform(test_df),
                               columns=preprocessor.get_feature_names_out())
    
    y_train = train_df[mappings['drugs']]
    y_test = test_df[mappings['drugs']]
    y_train_trans = y_train.copy()
    y_test_trans = y_test.copy()
    for drug in mappings['drugs']:
        y_train_trans.loc[:, drug] = y_train[drug].replace({ "CL0": "C0",                           
                                "CL1": "C0",
                                "CL2": "C0",
                                "CL3": "C1",
                                "CL4": "C1",
                                "CL5": "C2",
                                "CL6": "C2"})
        y_test_trans.loc[:, drug] = y_test[drug].replace({  "CL0": "C0",
                                "CL1": "C0",
                                "CL2": "C0",
                                "CL3": "C1",
                                "CL4": "C1",
                                "CL5": "C2",
                                "CL6": "C2"})
    
    X_test_enc.index = y_test.index
    X_train_enc.index = y_train.index
    
    train_transformed = pd.concat([X_train_enc, y_train_trans], axis=1, ignore_index=True)
    test_transformed = pd.concat([X_test_enc, y_test_trans], axis=1, ignore_index=True)
    
    train_transformed.columns = list( preprocessor.get_feature_names_out()) + list(mappings['drugs'])
    test_transformed.columns =  list(preprocessor.get_feature_names_out()) + list(mappings['drugs'])
    
    train_transformed.to_csv(os.path.join(processed_out_dir, "train.csv"), index=False)
    test_transformed.to_csv(os.path.join(processed_out_dir, "test.csv"), index=False)

if __name__ == "__main__":
    opt = docopt(__doc__)
    main(opt['--input_file_path'], 
         opt['--preprocessed_out_dir'], 
         opt['--processed_out_dir'])