# author: Jenit Jain, Shaun Hutchinson, Ritisha Sharma
# date: 2022-11-23

"""
Usage: preprocess.py --input_file_path=<input_file_path> --preprocessed_out_dir=<preprocessed_out_dir> --processed_out_dir=<processed_out_dir>
Options:
--input_file_path=<input_file_path>                 Takes the input file path for the original file (this is a required positional argument)
--preprocessed_out_dir=<preprocessed_out_dir>       Takes a directory path to dump the preprocessed data (this is a required positional argument)
--processed_out_dir=<processed_out_dir>             Takes a directory path to dump the processed and encoded data (this is a required positional argument)
"""

import os
from docopt import docopt
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import json

#The following JSON file contains all the mappings required for data-cleaning 
mappings = json.load(open("src/data_mapping.json", "r"))

def main(in_path, preprocessed_out_dir, processed_out_dir):
    # df = pd.read_table("../data/raw/drug_consumption.data", index_col=0, names=column_names, delimiter=',')
    df = pd.read_table(in_path, index_col=0, names=mappings["column_headers"], delimiter=',')

    for key, values in mappings["categories"].items():
        #The float values are keys and are henced saved as string values in the JSON object
        #Hence we are casting them back to float here
        values = {float(k):v for k,v in values.items()}
        df[key] = df.replace({key:values})[key]

    # df.to_csv("../data/preprocessed/drug_consumption.csv", index=False)
    train_df, test_df = train_test_split(df, train_size=0.8, shuffle=False, random_state=522)
    
    train_df.to_csv(os.path.join(preprocessed_out_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(preprocessed_out_dir, "test.csv"), index=False)
    
    #Perform col transformation
    preprocessor =  make_column_transformer(
                        (StandardScaler(), mappings['numerical']),
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
    
    train_transformed = pd.concat([X_train_enc, y_train], axis=1)
    test_transformed = pd.concat([X_test_enc, y_test], axis=1)
    
    train_transformed.to_csv(os.path.join(processed_out_dir, "train.csv"), index=False)
    test_transformed.to_csv(os.path.join(processed_out_dir, "test.csv"), index=False)

if __name__ == "__main__":
    opt = docopt(__doc__)
    main(opt['--input_file_path'], 
         opt['--preprocessed_out_dir'], 
         opt['--processed_out_dir'])
    
    
    