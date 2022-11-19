# Drug Consumption Prediction
***

**Contributors/Authors**

- Shaun Hutchinson
- Jenit Jain
- Ritisha Sharma

### About
***

The objective of this project is to predict if a person will have never used a drug, used over it a decade ago, used it in the last decade, used it in the last year, used it in the last month, used it in the last week, and used it in the last day regarding a selection of drugs given their personality measurements, NEO-FFI-R (neuroticism, extraversion, openness to experience, agreeableness, and conscientiousness), BIS-11 (impulsivity), and ImpSS (sensation seeking), and personal characteristics (level of education, age, gender, country of residence, ethnicity). We will attempt to find the most suitable classification model by testing decision trees and other models of increasing complexity. One EDA would be comparing the distribution of personality measurements based on the reported use of a certain drug. The EDA will be displayed using line charts with drug usage frequencies plots, and additionally boxplots containing the distribution for the selected personality measurements. The results of our analysis would be displayed using multiple violin plots, each containing the distribution for the selected personality measurements.

The dataset for this project is created by Elaine Fehrman, Vincent Egan, Evgeny Mirkes. It contains reponses from 1885 people about their usage of 18 different drugs, their personality measurements and other characteristics.


## Usage
# Downloading Data
In order to use run the anlysis, you can download the data using the script located ([here](https://github.com/UBC-MDS/drug_consumption_prediction/blob/download_data/src/download_data.py)).

# EDA
The exploritory data analysis can be replicated using the following .ipnyb located ([here](https://github.com/UBC-MDS/drug_consumption_prediction/blob/EDA/src/drug_consumption_data_analysis.ipynb)). In order run this anlysis, install the dependencies below.
## Dependencies
- Python 3.10.6 and Python packages:
    - docopt-ng = 0.8.1
    - altair 4.2.0
    - pandas = 1.4.3
    - numpy 1.23.3
    - hpsklearn 1.0.3
### Licenses
***
All material for this project is made available under the **Creative Commons Attribution 4.0 Canada License** ([CC BY 4.0 CA](https://creativecommons.org/licenses/by-nc-nd/4.0/)).

Except where otherwise noted, the example programs and other software
provided in the introduction-to-data-science repository are made available under the
MIT license.

### References
***
Fehrman,Elaine, Egan,Vincent & Mirkes,Evgeny. (2016). Drug consumption (quantified). UCI Machine Learning Repository.