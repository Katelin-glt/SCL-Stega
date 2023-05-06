import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

ours = [0.9783,0.9558,0.9295,0.9103,0.8648]
ls_cnn = [0.9481, 0.8908,0.8484,0.8093,0.7614]
dense_lstm = [0.9267,0.8911,0.8471,0.7768,0.7561]
bert = [0.9553,0.9306,0.8869,0.8570,0.8097]
sesy = [0.9580,0.9123,0.8659,0.8079,0.7850]

# ours = [0.9915,0.9743,0.9590,0.9330,0.8928]
# ls_cnn = [0.9724,0.9369,0.9018,0.8498,0.8024]
# dense_lstm = [0.9645,0.9203,0.8820,0.8340,0.7722]
# bert = [0.9821,0.9623,0.9293,0.8992,0.8594]
# sesy = [0.9695,0.9411,0.9038,0.8600,0.8168]

input_values = [0.939, 1.737, 2.381, 3.036, 3.619]
# input_values = [0.601, 1.420, 2.144, 2.902, 3.313]


fig, ax = plt.subplots()  # fig表示整张图片，ax表示图片中的各个图表
ax.set_title("Twitter-VLC", fontsize=20)
ax.set_xlabel("bpw", fontsize=15)
ax.set_ylabel("Accuracy", fontsize=15)

ax.plot(input_values, ls_cnn, marker='o', label=u'LS-CNN+CE')
ax.plot(input_values, dense_lstm, marker='x', label=u'Dense+CE')
ax.plot(input_values, bert, marker='+', label=u'BERT+CE')
ax.plot(input_values, sesy, marker='v', label=u'Sesy-GAT+CE')
ax.plot(input_values, ours, marker='*', label=u'SCL-Stega')  # 横坐标数据+纵坐标数据+图例
plt.xticks(input_values)
plt.legend(loc="lower left")
# 添加网格线
plt.grid(True, alpha=0.5, axis='both', linestyle=':')
fig.savefig("Twitter-HC-line-new.png", dpi=500);
plt.show()