import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def confusion_matrix(temp, lam, acc):
    df = pd.DataFrame(acc, index=temp, columns=lam)
    sns.heatmap(df, annot=True,fmt=".4f",
                vmin=0.9845, vmax=0.9895, cmap="GnBu")
    plt.xlabel('lamda')
    plt.ylabel('temperature')
    plt.savefig("Hyperparameter_grid_search_new.pdf", dpi=750, bbox_inches='tight')
    plt.show()
#
temp = ["0.1", "0.3", "0.5", "0.7", "0.9"]
lam = ["0.1", "0.3", "0.5", "0.7", "0.9"]
acc = [[0.9850,0.9878,0.9875,0.9890,0.9870],
       [0.9855,0.9890,0.9868,0.9868,0.9863],
       [0.9863,0.9895,0.9868,0.9870,0.9870],
       [0.9853,0.9870,0.9875,0.9890,0.9860],
       [0.9875,0.9855,0.9845,0.9875,0.9875]]

confusion_matrix(temp, lam, acc)

