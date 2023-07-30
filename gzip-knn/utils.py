import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns 

from sklearn.metrics import confusion_matrix, classification_report, 

def plot_confusion_matrix(val_y, pred, classes, save_img):
    plt.figure(figsize = (15, 13), facecolor = 'silver', edgecolor = 'gray')

    cm = confusion_matrix(val_y, pred, labels=classes)

    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(classes); ax.yaxis.set_ticklabels(classes)
    plt.savefig(save_img)


def plot_classification_report(val_y, pred, classes, save_img):
    plt.figure(figsize = (15, 15), facecolor = 'silver', edgecolor = 'gray')

    cr = classification_report(val_y, pred,
                                   target_names = classes,
                                   output_dict = True)
                                   
    sns.heatmap(pd.DataFrame(cr).iloc[:-1, :].T, annot=True)
    plt.savefig(save_img)
