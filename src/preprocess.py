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
from sklearn.model_selection import train_test_split
import json

#The following JSON file contains all the mappings required for data-cleaning 
mappings = json.load(open("data_mapping.json", "r"))

def main(in_path, preprocessed_out_dir, processed_out_dir):
    """
    Processes raw data into preprocessed and cleans to processed data
    Parameters
    ----------
    in_path : string
        path from where data is read
    preprocessed_out_dir: string
        path to the directory where preprocessed data is stored
    processed_out_dir: string
        path to the directory where processed data is stored
    Returns
    -------
    None
    Example
    --------
    main("../data/raw/drug_consumption.data", "../data/preocessed", "../data/processed")
    """
    # df = pd.read_table("../data/raw/drug_consumption.data", index_col=0, names=column_names, delimiter=',')
    df = pd.read_table(in_path, index_col=0, names=mappings["column_headers"], delimiter=',')

    for key, values in mappings["categories"].items():
        #The float values are keys and are henced saved as string values in the JSON object
        #Hence we are casting them back to float here
        values = {float(k):v for k,v in values.items()}
        df[key] = df.replace({key:values})[key]

    # df.to_csv("../data/preprocessed/drug_consumption.csv", index=False)
    train_df, test_df = train_test_split(df, train_size=0.8, shuffle=False, random_state=522)

    try:
        train_df.to_csv(os.path.join(preprocessed_out_dir, "train.csv"), index=False)
    except:
        os.makedirs(os.path.dirname(preprocessed_out_dir))
        train_df.to_csv(os.path.join(preprocessed_out_dir, "train.csv"), index=False)

    try:
        test_df.to_csv(os.path.join(preprocessed_out_dir, "test.csv"), index=False)
    except:
        os.makedirs(os.path.dirname(preprocessed_out_dir))
        test_df.to_csv(os.path.join(preprocessed_out_dir, "test.csv"), index=False)
    
    # Drop drugs columns that will not be used for analysis
    train_df = train_df.drop(columns = mappings["drop"])
    test_df = test_df.drop(columns = mappings["drop"])

    # Clean up strings in categorical and ordinal columns
    for col_name in mappings["categorical"] + mappings["ordinal"]:
        train_df[col_name] = train_df[col_name].str.strip()
        test_df[col_name] = test_df[col_name].str.strip()

    try:
        train_df.to_csv(os.path.join(processed_out_dir, "train.csv"), index=False)
    except:
        os.makedirs(os.path.dirname(processed_out_dir))
        train_df.to_csv(os.path.join(processed_out_dir, "train.csv"), index=False)

    try:
        test_df.to_csv(os.path.join(processed_out_dir, "test.csv"), index=False)
    except:
        os.makedirs(os.path.dirname(processed_out_dir))
        test_df.to_csv(os.path.join(processed_out_dir, "test.csv"), index=False)

if __name__ == "__main__":
    opt = docopt(__doc__)
    main(opt['--input_file_path'], 
         opt['--preprocessed_out_dir'], 
         opt['--processed_out_dir'])
    
    
    