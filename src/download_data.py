"""This script downloads data into a csv file given a URL and local file path.

Usage: download_data.py --url=<url> --file_path=<file_path>

Example: python src/download_data.py --url=https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data --file_path=data/raw

Options:
--url=<url>                 Takes in the url (this is a required option)
--file_path=<file_path>     Takes in the local file path (this is a required option)
""" 

from docopt import docopt
import pandas as pd
import os

def main(url, file_path):
    """
    Downloads data from url into csv
    Parameters
    ----------
    url : string
        url from where data is read
    file_path: string
        path to which csv file is saved
    Returns
    -------
    None
    Example
    --------
    main("https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data", "data/raw")
    """
    data = pd.read_csv(url, header=None)
    # Save results to result path
    try:
        data.to_csv(os.path.join(file_path , "drug_consumption.csv"), index = False, header = None)
    except:
        os.makedirs(file_path)
        data.to_csv(os.path.join(file_path , "drug_consumption.csv"), index = False, header = None)

if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments["--url"], arguments["--file_path"])
