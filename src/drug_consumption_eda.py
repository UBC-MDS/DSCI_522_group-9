"""Creates eda plots and tables from the pre-processed training data which was optained the drug consumption dataset (from https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data).
Saves the plots as png files and the tables as csv files.

Usage: src/drug_consumption_eda.py --train=<train> --out_dir=<out_dir>

Example: python src/drug_consumption_eda.py --train=data/preprocessed/train.csv --out_dir=results/eda

Options:
--train=<train>     Path (including filename) to training data (which needs to be saved as a CSV)
--out_dir=<out_dir> Path to directory where the plots should be saved
"""

import altair as alt
import pandas as pd
from sklearn.model_selection import train_test_split
from docopt import docopt
import pandas as pd
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def main(train, out_dir):
    train_df = pd.read_csv(train)
    # Making frequency of drug consumption chart
    
    feature_columns = [ 'Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Neuroticism',
                    'Extraversion', 'Openness', 'Agreeableness', 'Conscientiousness',
                    'Impulsiveness', 'SensationSeeking']

    target_columns = [  'Chocolate', 'Caffeine',  'Nicotine', 'Alcohol', 
                    'Cannabis', 'Mushrooms','Cocaine', 'VSA']

    class_df = train_df.melt(
        id_vars=feature_columns,
        value_vars = target_columns,
        var_name = "Drug",
        value_name = 'Class'
    )

    class_df['Class'] = class_df['Class'].str.replace("CL0", "0")
    class_df['Class'] = class_df['Class'].str.replace("CL1", "1")
    class_df['Class'] = class_df['Class'].str.replace("CL2", "2")
    class_df['Class'] = class_df['Class'].str.replace("CL3", "3")
    class_df['Class'] = class_df['Class'].str.replace("CL4", "4")
    class_df['Class'] = class_df['Class'].str.replace("CL5", "5")
    class_df['Class'] = class_df['Class'].str.replace("CL6", "6")

    class_df["Class"].astype(int)

    frequency_order = ["Never Used",
                    "Used over a Decade Ago",
                    "Used in Last Decade",
                    "Used in Last Year",
                    "Used in Last Month",
                    "Used in Last Week",
                    "Used in Last Day"]
    axis_labels = ""
    
    for i in range(len(frequency_order)):
        axis_labels += f"datum.label == {i} ? '{frequency_order[i]}' : "
    axis_labels += "'Unknown'"

    select_drug = alt.selection_multi(fields=['Drug'], bind='legend')

    point_chart = alt.Chart(class_df).mark_point(filled=True).encode(
        x = alt.X("Class:O", 
                axis=alt.Axis(labelExpr=axis_labels, labelAngle=0), 
                title="Frequency of Drug consumption"),
        y = alt.Y("count()", title='Number of people'),
        tooltip='Drug',
        color = "Drug",
        opacity=alt.condition(select_drug, alt.value(0.7), alt.value(0.1))
    ).properties(
        title="Frequency of drug consumption for different drugs",
        width = 800,
        height = 500
    )

    drug_frequency = (point_chart + point_chart.mark_line()).add_selection(select_drug)
    filepath = os.path.join(out_dir , "drug_frequency.png")
    drug_frequency.save(filepath)
    
    # Making Personality Score Chart 
    
    personality_df = train_df.melt(
    id_vars= ['Age', 'Gender', 'Education', 'Country', 'Ethnicity'],
    value_vars = ['Neuroticism', 'Extraversion', 'Openness', 'Agreeableness', 'Conscientiousness'],
    var_name = "Measure",
    value_name = 'Value'
    )   
    personality_chart = alt.Chart(personality_df).mark_boxplot().encode(
    x = alt.X('Value', scale=alt.Scale(zero=False), title='Score'),
    y = alt.Y("Measure", title="Personality Measure"),
    color = alt.Color("Measure",legend=None)
    ).properties(
    title='Spread of scores obtained from the NEO-FFI-R test',
    width=400,
    height=150
    )
    filepath = os.path.join(out_dir , "personality_chart.png")
    drug_frequency.save(filepath)
    personality_chart.save(filepath)
    
    # Making value count tables
    categorical_features = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity']
    for feat in categorical_features:
        df = train_df.groupby(feat)[feat].agg(['count'])
        df['Proportion'] = round(df['count'] / df['count'].sum() * 100, 2)
        table_name = '_'.join((feat, "valuecount.csv"))
        filepath = os.path.join(out_dir , table_name)
        df.to_csv(filepath)
    
    # Numerical bar plots
    imp_chart = alt.Chart(train_df).mark_bar().encode(
    x=alt.X("Impulsiveness", sort=[f"C{i}" for i in range(1,11)], title=""),
    y=alt.Y("count()")
    ).properties(
        width=300,
        title='Distribution of the Impulsiveness score'
    )
    ss_chart = alt.Chart(train_df).mark_bar().encode(
        x=alt.X("SensationSeeking", sort=[f"C{i}" for i in range(1,12)], title="Score"),
        y="count()"
    ).properties(
        width=300,
        title='Distribution of the Sensation Seeking score'
    )
    numerical_bars = imp_chart & ss_chart
    filepath = os.path.join(out_dir , "numerical_bars.png")
    numerical_bars.save(filepath)
    
if __name__ == "__main__":    
    opt = docopt(__doc__)
    main(opt["--train"], opt["--out_dir"])
    

    