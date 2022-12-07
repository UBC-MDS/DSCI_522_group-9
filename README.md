# Drug Consumption Prediction
***

**Contributors/Authors**

- Shaun Hutchinson
- Jenit Jain
- Ritisha Sharma

## About

The objective of this project is to predict the level of consumption of a selection of drugs given their personality measurements, NEO-FFI-R (neuroticism, extraversion, openness to experience, agreeableness, and conscientiousness), BIS-11 (impulsivity), and ImpSS (sensation seeking), and personal characteristics (level of education, age, gender, country of residence, ethnicity). 

For this model will attempt to predict the classification using SVM RBF classification model . One EDA would be comparing the distribution of personality measurements based on the reported use of a certain drug. 

The dataset for this project is created by Elaine Fehrman, Vincent Egan, Evgeny Mirkes. It contains reponses from 1885 people about their usage of 18 different drugs, their personality measurements and other characteristics.

## Usage
***
## Downloading Data
In order to use run the anlysis, you can download the data using the script located ([here](https://github.com/UBC-MDS/drug_consumption_prediction/blob/download_data/src/download_data.py)). The dataset is located at the following URL: https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data. This data should be stored in the following file path `data/raw/`. The arguments being:
 --url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data" 
 -- file_path= `"data/raw/"`

## Preprossesing
The raw data requires some cleaning and can be replicated using the following script located ([here](https://github.com/UBC-MDS/drug_consumption_prediction/blob/main/src/preprocess.py)). In order run this document, install the dependencies below.
The arguments for this script should be:
--input_file="../data/raw/drug_consumption.data"
--preproccessed_out_dir="./data/preprocessed"
--processed_out_dir="../data/processed"
## EDA
The exploritory data analysis can be replicated using the following script located ([here](https://github.com/UBC-MDS/drug_consumption_prediction/blob/main/src/drug_consumption_eda.py)). In order run this anlysis, install the dependencies below.
The arguments for this script should be:
--train="../data/preprocessed/train.csv"
--out_dir="../results/eda/"

## Model
The SVM RBF Model analysis can be replicated using the following script located ([here](https://github.com/UBC-MDS/drug_consumption_prediction/blob/main/src/drug_consumption_prediction_model.py)). In order run this anlysis, install the dependencies below.
The arguments for this script should be:
--data_path="../data/preprocessed/"
--result_path=""../results/"

## Makefile
To reset the repository to the initial state, with no intermediate or results files, run the following command  in the terminal from the root directory of this project:
> make clean

To replicate all of the analysis, run the following command in the terminal from the root directory of this project:
> make all

## Dependencies
- Python 3.10.6 and Python packages:
    - docopt-ng = 0.8.1
    - altair 4.2.0
    - pandas = 1.4.3
    - numpy 1.23.3
    - hpsklearn 1.0.3
    - sklearn = 1.1.2
    - dataframe_image = 0.1.1
- R version 4.2.1
    - knitr = 1.40
    - kableExtra = 1.3.4
    - tidyverse = 1.3.2
    - caret = 6.0.93
## Licenses
All material for this project is made available under the **Creative Commons Attribution 4.0 Canada License** ([CC BY 4.0 CA](https://creativecommons.org/licenses/by-nc-nd/4.0/)).

Except where otherwise noted, the example programs and other software
provided in the introduction-to-data-science repository are made available under the
MIT license.

This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.

This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.

# References
***
Fehrman,Elaine, Egan,Vincent & Mirkes,Evgeny. (2016). Drug consumption (quantified). UCI Machine Learning Repository.
