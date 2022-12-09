# author: Jenit Jain, Shaun Hutchinson, Ritisha Sharma
# 2021-11-26

"""This script takes in training and testing set of drug consumption data set, 
fits an SVC model on the training set and evaluates on the testing set.

Usage: drug_consumption_prediction_model.py --data_path=<data_path> --result_path=<result_path>

Example: python src/drug_consumption_prediction_model.py --data_path=data/preprocessed --result_path=results/analysis

Options:
--data_path=<data_path>         Takes in the path to the data (this is a required option)
--result_path=<result_path>     Takes in the file path to save the resulting figures/tables (this is a required option)
""" 

from docopt import docopt
import pandas as pd
import numpy as np
import dataframe_image as dfi
import os
import json

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
    OneHotEncoder,
    PolynomialFeatures
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate

from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def create_preprocessor():
    """
    Created a preprocessor specific to this analysis and dataset

    Parameters
    ----------
    None

    Returns
    -------
    preprocessor : ColumnTransformer
        Contains the transformers for each column-type

    Example
    --------
    create_preprocessor()
    """
    mappings = json.load(open("src/data_mapping.json", "r"))
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
    return preprocessor

def train_model(model, param_dist, X, y):
    """
    Uses the model provided in the arguments with the parameter distribution to find best scores after optimization

    Parameters
    ----------
    model : scikit model
        The model to be fit and evaluated
    param_dist : dictionary
        The hyperparameter that model is optimized over
    X : np.array
        The training data without targets
    y : np.array
        The targets for training data

    Returns
    -------
    model_best_estimator : scikit model
        The best model found by RandomizedSearchCV
    score_by_drug : pd.DataFrame
        Contains the scores according to the drug

    Example
    --------
    train_model(svc, param_dist, X_train, y_train)
    """
    mappings = json.load(open("src/data_mapping.json", "r"))
    drug_columns = mappings['drugs']

    ## Fit model ---------------------------------------------
    # Save the best model and score for each drug
    model_best_estimator = {}
    model_best_score_by_drug = {}
    for drug in drug_columns: 
        random_search = RandomizedSearchCV(
            model, param_distributions = param_dist, n_jobs = -1, n_iter = 10, cv = 2, 
            return_train_score = True, random_state = 522
        )
        random_search.fit(X, y[drug])
        model_best_estimator[drug] = random_search.best_estimator_
        model_best_score_by_drug[drug] = [round(random_search.best_score_, 4)]
        
    # Save best scores to dataframe
    score_by_drug = pd.DataFrame(model_best_score_by_drug).T
    score_by_drug = score_by_drug.reset_index()
    
    return model_best_estimator, score_by_drug

def get_dt_feature_importances(X_train, y_train):
    """
    Uses a Decision Tree on the training data to get feature importances

    Parameters
    ----------
    X_train : np.array
        The training data without targets
    y_train : np.array
        The targets for training data

    Returns
    -------
    feature_importance_drug : pd.DataFrame
        Contains the calculated feature importances

    Example
    --------
    get_dt_feature_importances(X_train, y_train)
    """
    preprocessor = create_preprocessor()

    mappings = json.load(open("src/data_mapping.json", "r"))
    drug_columns = mappings['drugs']
    # Make column transformer
    
    drop_columns = mappings['drop']
    drop_columns.remove("Country")
    
    tree_preprocessor =  make_column_transformer(
        (StandardScaler(), mappings['numerical']),
        (OrdinalEncoder(categories = [
                            list(mappings['categories']['Age'].values()),
                            list(mappings['categories']['Education'].values()),
                            list(mappings['categories']['Impulsiveness'].values()),
                            list(mappings['categories']['SensationSeeking'].values()),
                        ]), mappings['ordinal']),
        (OneHotEncoder(drop='if_binary', dtype=int, handle_unknown='ignore'), mappings['categorical'] + ['Country']),
        ("drop", drop_columns)
    )
     
    # Look at feature importances with decision tree   
    tree_clf_pipe =  make_pipeline(
        tree_preprocessor, 
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
    # Add highlighting to distinguish between importances
    feature_importance_drug = feature_importance_drug.set_index("feature").style.background_gradient(cmap = "BuPu")

    return feature_importance_drug

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

    # Read mappings
    mappings = json.load(open("src/data_mapping.json", "r"))
    # Separate out target columns
    drug_columns = mappings['drugs']

    # Separate X and y 
    X_train = train_df.drop(columns = drug_columns)
    y_train = train_df[drug_columns]
    X_test = test_df.drop(columns = drug_columns)
    y_test = test_df[drug_columns]
    
    for drug in mappings['drugs']:
        y_train[drug].replace({ "CL0": "C0",                           
                                "CL1": "C0",
                                "CL2": "C0",
                                "CL3": "C1",
                                "CL4": "C1",
                                "CL5": "C2",
                                "CL6": "C2"},
                              inplace=True)
    
        y_test[drug].replace({  "CL0": "C0",
                                "CL1": "C0",
                                "CL2": "C0",
                                "CL3": "C1",
                                "CL4": "C1",
                                "CL5": "C2",
                                "CL6": "C2"},
                             inplace=True)
    
    ## Get baseline scores ---------------------------------------------
    # DummyClassifier
    dc = DummyClassifier()

    dummy_cv_results = {}
    # Get the mean accuracy for each drug
    for drug in drug_columns: 
        dummy_cv_results[drug] = pd.DataFrame(cross_validate(dc, X_train, y_train[drug], cv = 2,
                                                return_train_score = True)).mean().round(4)
    
    # Save results in a DataFrame
    dummy_cv_results = pd.DataFrame(dummy_cv_results)
    dummy_cv_results = dummy_cv_results.drop(index = ["fit_time", "score_time"]).T
    dummy_cv_results = dummy_cv_results.reset_index()
    dummy_cv_results = dummy_cv_results.rename(columns = {"index": "target_drug"})
    
    preprocessor = create_preprocessor()
    
    # SVC -- hyperparameter optimization
    svc_pipe =  make_pipeline(
        preprocessor, 
        SVC()
    )

    param_dist = {"svc__class_weight": ["balanced", None],
            "svc__gamma": 10.0 ** np.arange(-4, 4),
             "svc__C": 10.0 ** np.arange(-4, 4)}

    svc_best_estimator, score_by_drug = train_model(svc_pipe, param_dist, X_train, y_train)
    score_by_drug = score_by_drug.rename(columns = {"index": "target_drug", 0: "svc_score"})
    score_by_drug["dummy_score"] = dummy_cv_results["test_score"]
    
    # Save results to result path
    results_path = os.path.join(result_path, "svc_dummy_score.csv")
    try:
        score_by_drug.to_csv(results_path, index = False)
    except:
        os.makedirs(os.path.dirname(results_path))
        score_by_drug.to_csv(results_path, index = False)
    
    feature_importance_drug = get_dt_feature_importances(X_train, y_train)

    
    # Save png to result path
    fi_path = os.path.join(result_path, "feature_importances.png")
    try:
        dfi.export(feature_importance_drug, fi_path, table_conversion='matplotlib')
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
