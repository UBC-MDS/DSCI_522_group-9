# data pipe - drug consumption prediction
# author: Shaun Hutchinson, Jenit Jain, Ritisha Sharma
# date: 2022-11-29

all: doc/drug_consumption_prediction_report.html

# download and save data from url
data/raw/drug_consumption.csv: src/download_data.py
	python src/download_data.py --url=https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data --file_path=data/raw

# preprocess data (split data and clean strings)
data/preprocessed/train.csv data/preprocessed/test.csv data/processed/train.csv data/processed/test.csv: src/preprocess.py data/raw/drug_consumption.csv
	python src/preprocess.py --input_file_path=data/raw/drug_consumption.csv --preprocessed_out_dir=data/preprocessed --processed_out_dir=data/processed 

# exploratory data analysis (frequency chart, personality scores chart, value counts table, numerical features barplot)
results/eda/drug_frequency.png results/eda/personality_chart.png results/eda/numerical_bars.png results/eda/Age_valuecount.png results/eda/Gender_valuecount.png results/eda/Education_valuecount.png results/eda/Country_valuecount.png results/eda/Ethnicity_valuecount.png: src/drug_consumption_eda.py data/preprocessed/train.csv
	python src/drug_consumption_eda.py --train=data/preprocessed/train.csv --out_dir=results/eda

# train model using SVC and assess on testing data
results/analysis/feature_importances.png results/analysis/svc_dummy_score.csv results/analysis/test_results.csv: src/drug_consumption_prediction_model.py data/processed/train.csv data/processed/test.csv
	python src/drug_consumption_prediction_model.py --data_path=data/processed --result_path=results/analysis

# render Rmarkdown report
doc/drug_consumption_prediction_report.html: doc/drug_consumption_prediction_report.Rmd doc/drug_prediction_refs.bib results/eda/drug_frequency.png results/eda/personality_chart.png results/eda/numerical_bars.png results/eda/Age_valuecount.png results/eda/Gender_valuecount.png results/eda/Education_valuecount.png results/eda/Country_valuecount.png results/eda/Ethnicity_valuecount.png  results/analysis/feature_importances.png results/analysis/svc_dummy_score.csv results/analysis/test_results.csv
	Rscript -e "rmarkdown::render('doc/drug_consumption_prediction_report.Rmd', output_format = 'html_document')"

clean: 
	rm -rf data
	rm -rf results/eda/*
	rm -rf results/analysis/*
	rm -rf doc/drug_consumption_prediction_report.html
