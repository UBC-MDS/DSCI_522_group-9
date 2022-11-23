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


column_names = ["Age", 
                "Gender", 
                "Education",
                "Country",
                "Ethnicity",
                "Neuroticism",
                "Extraversion",
                "Openness",
                "Agreeableness",
                "Conscientiousness",
                "Impulsiveness",
                "SensationSeeking",
                "Alcohol",
                "Amphetamines",
                "Amyl",
                "Benzos",
                "Caffeine",
                "Cannabis",
                "Chocolate",
                "Cocaine",
                "Crack",
                "Ecstacy",
                "Heroin",
                "Ketamine",
                "Legalh",
                "LSD",
                "Meth",
                "Mushrooms",
                "Nicotine",
                "Semer",
                "VSA"]

numerical_cols = ["Neuroticism",
                "Extraversion",
                "Openness",
                "Agreeableness",
                "Conscientiousness"]

ordinal_cols = ["Age", 
                "Education",
                "Impulsiveness",
                "SensationSeeking"]

categorical_cols = ["Gender", 
                "Country"]

drop_cols = ["Ethnicity",
             "Amphetamines",
             "Amyl",
             "Benzos",
             "Crack",
             "Ecstacy",
             "Heroin",
             "Ketamine",
             "Legalh",
             "LSD",
             "Meth",
             "Semer",
             ]

drug_classes = ["Alcohol",
                "Cannabis",
                "Chocolate",
                "Caffeine",
                "Cocaine",
                "Mushrooms",
                "Nicotine",
                "VSA"]

mappings = {
    "Age": {
        -0.95197: "18-24",
        -0.07854: "25-34",
        0.49788: "35-44",
        1.09449: "45-54",
        1.82213: "55-64",
        2.59171: "65+"
    },
    "Gender": {0.48246: "Female", -0.48246: "Male"},
    "Education": {
        -2.43591: "Left school before 16 years ",
        -1.7379: "Left school at 16 years ",
        -1.43719: "Left school at 17 years ",
        -1.22751: "Left school at 18 years ",
        -0.61113: "Some college or university, no certificate or degree ",
        -0.05921: "Professional certificate/ diploma",
        0.45468: "University degree ",
        1.16365: "Masters degree ",
        1.98437: "Doctorate degree"
    },
    "Country": {
        -0.09765: "Australia",
        0.24923: "Canada",
        -0.46841: "New Zealand",
        -0.28519: "Other",
        0.21128: "Republic of Ireland",
        0.96082: "UK",
        -0.57009: "USA"
    },
    "Ethnicity": {
        -0.50212: "Asian",
        -1.10702: "Black",
        1.90725: "Mixed-Black/Asian",
        0.126: "Mixed-White/Asian",
        -0.22166: "Mixed-White/Black",
        0.1144: "Other",
        -0.31685: "White"
    },
    "Neuroticism": {
        -3.46436: 12,
        -0.67825: 29,
        1.02119: 46,
        -3.15735: 13,
        -0.58016: 30,
        1.13281: 47,
        -2.75696: 14,
        -0.46725: 31,
        1.13: 47,
        1.37: 40,
        1.49: 50,
        1.6: 51,
        1.84: 53,
        1.23461: 48,
        -2.52197: 15,
        -2.52: 15,
        -0.34799: 32,
        1.37297: 49,
        -2.42317: 16,
        -0.24649: 33,
        1.49158: 50,
        -2.3436: 17,
        -2.34: 17,
        -0.14882: 34,
        1.60383: 51,
        -2.21844: 18,
        -0.05188: 35,
        1.72012: 52,
        -2.05048: 19,
        0.04257: 36,
        1.8399: 53,
        -1.86962: 20,
        0.13606: 37,
        1.98437: 54,
        -1.69163: 21,
        0.22393: 38,
        2.127: 55,
        -1.55078: 22,
        0.31287: 39,
        2.28554: 56,
        -1.43907: 23,
        0.41667: 40,
        2.46262: 57,
        -1.32828: 24,
        0.52135: 41,
        2.61139: 58,
        -1.1943: 25,
        0.62967: 42,
        2.82196: 59,
        -1.05308: 26,
        0.73545: 43,
        3.27393: 60,
        -0.92104: 27,
        0.82562: 44,
        -0.79151: 28,
        0.91093: 45},
    "Extraversion": {
        -3.27393: 16,
        -1.23177: 31,
        0.80523: 45,
        -3.00537: 18,
        -1.09207: 32,
        0.96248: 46,
        -2.72827: 19,
        -0.94779: 33,
        1.11406: 47,
        -2.5383: 20,
        -0.80615: 34,
        1.2861: 48,
        -2.44904: 21,
        -0.69509: 35,
        1.45421: 49,
        -2.32338: 22,
        -0.57545: 36,
        1.58487: 50,
        -2.21069: 23,
        -0.43999: 37,
        1.74091: 51,
        -2.11437: 24,
        -0.30033: 38,
        1.93886: 52,
        -2.03972: 25,
        -0.15487: 39,
        2.127: 53,
        -1.92173: 26,
        0.00332: 40,
        2.32338: 54,
        -1.7625: 27,
        0.16767: 41,
        2.57309: 55,
        -1.6334: 28,
        0.32197: 42,
        2.8595: 56,
        -1.50796: 29,
        0.47617: 43,
        3.00537: 58,
        -1.37639: 30,
        0.63779: 44,
        3.27393: 59
    },
    "Openness": {
        -3.27393: 24,
        -1.11902: 38,
        0.58331: 50,
        -2.8595: 26,
        -0.97631: 39,
        0.7233: 51,
        -2.63199: 28,
        -0.84732: 40,
        0.88309: 52,
        -2.39883: 29,
        -0.71727: 41,
        1.06238: 53,
        -2.21069: 30,
        -0.58331: 42,
        1.24033: 54,
        -2.09015: 31,
        -0.45174: 43,
        1.43533: 55,
        -1.97495: 32,
        -0.31776: 44,
        1.65653: 56,
        -1.82919: 33,
        -0.17779: 45,
        1.88511: 57,
        -1.68062: 34,
        -0.01928: 46,
        2.15324: 58,
        -1.55521: 35,
        0.14143: 47,
        2.44904: 59,
        -1.42424: 36,
        0.29338: 48,
        2.90161: 60,
        -1.27553: 37,
        0.44585: 49
    },
    "Agreeableness": {
        -3.46436: 12,
        -1.34289: 34,
        0.76096: 48,
        -3.15735: 16,
        -1.21213: 35,
        0.94156: 49,
        -3.00537: 18,
        -1.07533: 36,
        1.11406: 50,
        -2.90161: 23,
        -0.91699: 37,
        1.2861: 51,
        -2.78793: 24,
        -0.76096: 38,
        1.45039: 52,
        -2.70172: 25,
        -0.60633: 39,
        1.61108: 53,
        -2.5383: 26,
        -0.45321: 40,
        1.81866: 54,
        -2.35413: 27,
        -0.30172: 41,
        2.03972: 55,
        -2.21844: 28,
        -0.15487: 42,
        2.23427: 56,
        -2.07848: 29,
        -0.01729: 43,
        2.46262: 57,
        -1.92595: 30,
        0.13136: 44,
        2.75696: 58,
        -1.772: 31,
        0.28783: 45,
        3.15735: 59,
        -1.6209: 32,
        0.43852: 46,
        3.46436: 60,
        -1.47955: 33,
        0.59042: 47
    },
    "Conscientiousness": {
        -3.46436: 17,
        -1.25773: 32,
        0.58489: 46,
        -3.15735: 19,
        -1.13788: 33,
        0.7583: 47,
        -2.90161: 20,
        -1.0145: 34,
        0.93949: 48,
        -2.72827: 21,
        -0.89891: 35,
        1.13407: 49,
        -2.57309: 22,
        -0.78155: 36,
        1.30612: 50,
        -2.42317: 23,
        -0.65253: 37,
        1.46191: 51,
        -2.30408: 24,
        -0.52745: 38,
        1.63088: 52,
        -2.18109: 25,
        -0.40581: 39,
        1.81175: 53,
        -2.04506: 26,
        -0.27607: 40,
        2.04506: 54,
        -1.92173: 27,
        -0.14277: 41,
        2.33337: 55,
        -1.78169: 28,
        -0.00665: 42,
        2.63199: 56,
        -1.64101: 29,
        0.12331: 43,
        3.00537: 57,
        -1.5184: 30,
        0.25953: 44,
        3.46436: 59,
        -1.38502: 31,
        0.41594: 45
    },
    "Impulsiveness": {
        -2.55524: "C1",
        -1.37983: "C2",
        -0.71126: "C3",
        -0.21712: "C4",
        0.19268: "C5",
        0.52975: "C6",
        0.88113: "C7",
        1.29221: "C8",
        1.86203: "C9",
        2.90161: "C10"
    },
    "SensationSeeking": {
        -2.07848: "C1",
        -1.54858: "C2",
        -1.18084: "C3",
        -0.84637: "C4",
        -0.52593: "C5",
        -0.21575: "C6",
        0.07987: "C7",
        0.40148: "C8",
        0.76540: "C9",
        1.22470: "C10",
        1.92173: "C11"
    }
 }

def main(in_path, preprocessed_out_dir, processed_out_dir):
    # df = pd.read_table("../data/raw/drug_consumption.data", index_col=0, names=column_names, delimiter=',')
    df = pd.read_table(in_path, index_col=0, names=column_names, delimiter=',')

    for key, values in mappings.items():
        df[key] = df.replace({key:values})[key]

    # df.to_csv("../data/preprocessed/drug_consumption.csv", index=False)
    train_df, test_df = train_test_split(df, train_size=0.8, shuffle=False, random_state=522)
    
    train_df.to_csv(os.path.join(preprocessed_out_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(preprocessed_out_dir, "test.csv"), index=False)
    
    
    #Perform col transformation
    preprocessor =  make_column_transformer(
                        (StandardScaler(), numerical_cols),
                        (OneHotEncoder(drop='if_binary', dtype=int, handle_unknown='ignore'), categorical_cols),
                        ("drop", drop_cols)
                    )
    #Save the train and test in a new folder
    preprocessor.fit(train_df)
    
    X_train_enc = pd.DataFrame(preprocessor.transform(train_df),
                               columns=preprocessor.get_feature_names_out())
    X_test_enc = pd.DataFrame(preprocessor.transform(test_df),
                               columns=preprocessor.get_feature_names_out())
    y_train = train_df[drug_classes]
    y_test = test_df[drug_classes]
    
    train_transformed = pd.concat([X_train_enc, y_train], axis=1)
    test_transformed = pd.concat([X_test_enc, y_test], axis=1)
    
    train_transformed.to_csv(os.path.join(processed_out_dir, "train.csv"), index=False)
    test_transformed.to_csv(os.path.join(processed_out_dir, "test.csv"), index=False)

if __name__ == "__main__":
    opt = docopt(__doc__)
    main(opt['--input_file_path'], 
         opt['--preprocessed_out_dir'], 
         opt['--processed_out_dir'])
    
    
    