from sklearn import datasets
from sklearn import preprocessing
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json
import numpy as np
import tsneutil

with open('Twitter-HC-0.939bpw-electra-scl.json', encoding='utf-8') as file:
    result = json.load(file)
    print(result.get('true'))
    print(len(result.get('feature')))

    X, y = result.get('feature'), result.get('true')
    # X, y = datasets.load_digits(return_X_y=True)

    # t-SNE降维处理
    tsne = TSNE(n_components=2, verbose=1 ,random_state=42)
    result = tsne.fit_transform(X)

    # 归一化处理
    # scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    # result = scaler.fit_transform(result)

    # 颜色设置
    # colors = plt.cm.rainbow(np.linspace(0,1,len(y)))

    # 可视化展示
    plt.figure(figsize=(12, 12))
    # plt.title('IMDB-AC-3bpw')
    # plt.xlim((-1.1, 1.1))
    # plt.ylim((-1.1, 1.1))
    # for i in range(len(y)):
    #     plt.text(result[i,0], result[i,1], str(y[i]),
    #              color=colors[y[i]], fontdict={'weight': 'bold', 'size': 9})
    tsneutil.plot(result, y, colors=tsneutil.MOUSE_10X_COLORS)
    # plt.scatter(result[:,0], result[:,1], c=y, s=10)
    # plt.show()
