import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from dotenv import load_dotenv
import os
import argparse
from tqdm import tqdm
import multiprocessing
from sklearn.model_selection import train_test_split 
from utils import plot_classification_report, plot_confusion_matrix
from ncd import calculate_ncd_row
from functools import partial

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--n_neighbor", required=False, default=8)

opt = parser.parse_args()


data_path = str(os.getenv("DATA_PATH"))
conf_matrix_path = str(os.getenv("CONFUSSION_MATRIX_PATH"))
class_report_path = str(os.getenv("CLASS_REPORT_PATH"))

dataset_path = os.path.join(data_path, opt.dataset)
dataset = pd.read_csv(dataset_path)

train_x, val_x, train_y, val_y = train_test_split(dataset.text.values, dataset.target.values, test_size=0.33, random_state=42)


NUM_PROCESSES = multiprocessing.cpu_count() 
if __name__ == "__main__":

    with multiprocessing.Pool(NUM_PROCESSES) as pool:
        partial_calculate_ncd_row = partial(calculate_ncd_row, trainset=train_x)

        train_data = list(enumerate(train_x)) 
        train_results = list(tqdm(pool.imap(partial_calculate_ncd_row, train_data), total=len(train_data)))

        val_data = list(enumerate(val_x))
        val_results = list(tqdm(pool.imap(partial_calculate_ncd_row, val_data), total=len(val_data)))

    train_ncd = np.zeros((len(train_x), len(train_x)))
    val_ncd = np.zeros((len(val_x), len(train_x)))

    for i, row in train_results:
        train_ncd[i] = row

    for i, row in val_results:
        val_ncd[i] = row

    knn = KNeighborsClassifier(n_neighbors=int(opt.n_neighbor))
    knn.fit(train_ncd, train_y)
    preds = knn.predict(val_ncd)
    val_accuracy = knn.score(val_ncd, val_y)
    train_accuracy = knn.score(train_ncd, train_y)

    print("Train accuracy:", train_accuracy)
    print("Validation accuracy:", val_accuracy)
    classes = knn.classes_.tolist()
    plot_confusion_matrix(val_y, preds, classes,  conf_matrix_path)
    plot_classification_report(val_y, preds, classes, class_report_path)


