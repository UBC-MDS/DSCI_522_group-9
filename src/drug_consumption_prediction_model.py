# Ritish Sharma
# 2021-11-26

"""This script takes in training and testing set of drug consumption data set, 
fits an SVC model on the training set and evaluates on the testing set.
Usage: drug_consumption_prediction_model.py --data_path=<data_path> --result_path=<result_path>
Options:
--data_path=<data_path>         Takes in the path to the data (this is a required option)
--result_path=<result_path>     Takes in the file path to save the resulting figures/tables (this is a required option)
""" 

from docopt import docopt
import pandas as pd
import numpy as np
import dataframe_image as dfi
import os

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate

from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def main(data_path, result_path):
    """
    Fits training data to SVC model and optimizes hyperparameters and evaluates on the testing set

    Parameters
    ----------
    data_path : string
        path from where data is read
    result_path: string
        path to which results are stored

    Returns
    -------
    None

    Example
    --------
    main("../data/processed/", "../results")
    """
    # Get data from the given path
    train_path = os.path.join(data_path, "train.csv")
    test_path = os.path.join(data_path, "test.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # For testing -- should be in preprocessing file
    # train_df["Education"] = train_df["Education"].str.strip()
    # test_df["Education"] = test_df["Education"].str.strip()

    # Separate out target columns
    drug_columns = ['Chocolate', 'Caffeine',  'Nicotine', 'Alcohol', 
                  'Cannabis', 'Mushrooms','Cocaine', 'VSA']

    # Separate X and y 
    X_train = train_df.drop(columns = drug_columns)
    y_train = train_df[drug_columns]
    X_test = test_df.drop(columns = drug_columns)
    y_test = test_df[drug_columns]
    
    # Column Transformations
    numerical_cols = ["Neuroticism", "Extraversion", "Openness",
                "Agreeableness", "Conscientiousness"]
    
    categorical_cols = ["Gender", "Country"]

    ordinal_cols = ["Age", "Education", "Impulsiveness", "SensationSeeking"]
    
    # Specify the order for encoding the ordinal columns
    age_order = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    education_order = ["Left school before 16 years", "Left school at 16 years", 
                   "Left school at 17 years", "Left school at 18 years", 
                   "Some college or university, no certificate or degree", 
                   "Professional certificate/ diploma", "University degree",
                  "Masters degree", "Doctorate degree"]
    impulsiveness_order = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"]
    sensation_seeking_order = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11"]

    # For testing -- should be in preprocessing file
    # drop_cols = ["Ethnicity", "Amphetamines", "Amyl", "Benzos",
    #     "Crack", "Ecstacy", "Heroin", "Ketamine", "Legalh",
    #      "LSD", "Meth", "Semer"]

    # Make column transformer
    preprocessor =  make_column_transformer(
        (StandardScaler(), numerical_cols),
        (OrdinalEncoder(categories = [age_order, education_order, 
                                  impulsiveness_order, sensation_seeking_order]), ordinal_cols),
        (OneHotEncoder(drop='if_binary', dtype=int, handle_unknown='ignore'), categorical_cols),
        # ("drop", drop_cols),
        remainder = "passthrough"
    )
    
    ## Get baseline scores ---------------------------------------------
    # DummyClassifier
    dc = DummyClassifier()

    dummy_cv_results = {}
    # Get the mean accuracy for each drug
    for drug in drug_columns: 
        dummy_cv_results[drug] = pd.DataFrame(cross_validate(dc, X_train, y_train[drug], cv = 5,
                                                return_train_score = True)).mean().round(4)
    
    dummy_cv_results = pd.DataFrame(dummy_cv_results)
    dummy_cv_results = dummy_cv_results.drop(index = ["fit_time", "score_time"]).T
    dummy_cv_results = dummy_cv_results.reset_index()
    dummy_cv_results = dummy_cv_results.rename(columns = {"index": "target_drug"})
    
    # SVC -- hyperparameter optimization
    svc_pipe =  make_pipeline(
        preprocessor, 
        SVC()
    )

    param_dist = {"svc__class_weight": ["balanced", None],
            "svc__gamma": 10.0 ** np.arange(-4, 4),
             "svc__C": 10.0 ** np.arange(-4, 4)}

    ## Fit SVC model ---------------------------------------------
    # Save the best model and score for each drug
    svc_best_estimator = {}
    svc_best_score_by_drug = {}
    for drug in drug_columns: 
        random_search = RandomizedSearchCV(
            svc_pipe, param_distributions = param_dist, n_jobs = -1, n_iter = 10, cv = 5, 
            return_train_score = True, random_state = 522
        )
        random_search.fit(X_train, y_train[drug])
        svc_best_estimator[drug] = random_search.best_estimator_
        svc_best_score_by_drug[drug] = [round(random_search.best_score_, 4)]
        
    score_by_drug = pd.DataFrame(svc_best_score_by_drug).T
    score_by_drug = score_by_drug.reset_index()
    score_by_drug = score_by_drug.rename(columns = {"index": "target_drug", 0: "svc_score"})
    score_by_drug["dummy_score"] = dummy_cv_results["test_score"]
    
    # Save results to result path
    results_path = os.path.join(result_path, "svc_dummy_score.csv")
    try:
        score_by_drug.to_csv(results_path, index = False)
    except:
        os.makedirs(os.path.dirname(results_path))
        score_by_drug.to_csv(results_path, index = False)
    
    # Look at feature importances with decision tree
    tree_clf_pipe =  make_pipeline(
        preprocessor, 
        DecisionTreeClassifier(random_state=522)
    )

    drug_feature_importances = {}
    for drug in drug_columns: 
        tree_clf_pipe.fit(X_train, y_train[drug])
        drug_feature_importances[drug] = tree_clf_pipe.named_steps["decisiontreeclassifier"].feature_importances_.tolist()
    
    # Create the datafame to show feature importances
    feature_importance_drug = pd.DataFrame(drug_feature_importances)
    feature_importance_drug["feature"] = tree_clf_pipe[0].get_feature_names_out().tolist()
    feature_importance_drug["feature"] = feature_importance_drug["feature"].str.split("__", expand = True)[1]
    feature_importance_drug = feature_importance_drug.set_index("feature").style.background_gradient(cmap = "BuPu")
    
    # Save png to result path
    fi_path = os.path.join(result_path, "feature_importances.png")
    try:
        dfi.export(feature_importance_drug, fi_path)
    except:
        os.makedirs(os.path.dirname(fi_path))
        dfi.export(feature_importance_drug, fi_path)
    
    ## Evaluate on test set ---------------------------------------------
    test_scores = {}
    for drug in drug_columns:
        best_model = svc_best_estimator[drug]
        test_scores[drug] = [round(best_model.score(X_test, y_test[drug]), 4)]

    test_scores = pd.DataFrame(test_scores).T
    test_scores = test_scores.reset_index()
    test_scores = test_scores.rename(columns = {"index": "target_drug", 0: "svc_score"})
    
    # Save results to result path
    test_results_path = os.path.join(result_path, "test_results.csv")
    try:
        test_scores.to_csv(test_results_path, index = False)
    except:
        os.makedirs(os.path.dirname(test_results_path))
        test_scores.to_csv(test_results_path, index = False)
    
if __name__ == '__main__':
    opt = docopt(__doc__)
    main(opt["--data_path"], opt["--result_path"])
