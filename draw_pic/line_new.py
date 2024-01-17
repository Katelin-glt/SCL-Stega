import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# Twitter-VLC
# input_values = [0.939, 1.737, 2.381, 3.036, 3.619]
#
# ours = [0.9783, 0.9558, 0.9295, 0.9103, 0.8648]
# ls_cnn = [0.9481, 0.8908, 0.8484, 0.8093, 0.7614]
# dense_lstm = [0.9267, 0.8911, 0.8471, 0.7768, 0.7561]
# bert = [0.9553, 0.9306, 0.8869, 0.8570, 0.8097]
# sesy = [0.9580, 0.9123, 0.8659, 0.8079, 0.7850]

# IMDB-AC
# input_values = [0.601, 1.420, 2.144, 2.902, 3.313]
# ours = [0.9915,0.9743,0.9590,0.9330,0.8928]
# ls_cnn = [0.9724,0.9369,0.9018,0.8498,0.8024]
# dense_lstm = [0.9645,0.9203,0.8820,0.8340,0.7722]
# bert = [0.9821,0.9623,0.9293,0.8992,0.8594]
# sesy = [0.9695,0.9411,0.9038,0.8600,0.8168]

# News-VLC
input_values = [0.965, 1.754, 2.440, 3.006, 3.729]
ours = [0.9868,0.9770,0.9613,0.9468,0.9183]
ls_cnn = [0.9535,0.9190,0.8828,0.8419,0.8035]
dense_lstm = [0.9324,0.9091,0.8734,0.8299,0.7893]
bert = [0.9751,0.9535,0.9320,0.9046,0.8681]
sesy = [0.9451,0.9175,0.8933,0.8543,0.8302]

fig, ax = plt.subplots()  # fig表示整张图片，ax表示图片中的各个图表
ax.set_title("News-VLC", fontsize=15)
ax.set_xlabel("bpw", fontsize=14)
ax.set_ylabel("Acc", fontsize=14)

ax.plot(input_values, ls_cnn, marker='o', markersize=5, label=u'LS-CNN+CE', color='#45C1E2')
ax.plot(input_values, dense_lstm, marker='o', markersize=5, label=u'Dense+CE', color='#4D6471')
ax.plot(input_values, bert, marker='o', markersize=5, label=u'BERT+CE', color='#F9CE65')
ax.plot(input_values, sesy, marker='o', markersize=5, label=u'Sesy-GAT+CE', color='#6675C0')
ax.plot(input_values, ours, marker='o', markersize=5, label=u'SCL-Stega(ours)', color='#F793CC')  # 横坐标数据+纵坐标数据+图例

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)


plt.xticks(input_values)
plt.legend(loc="lower left" , frameon=False)
# 添加网格线
plt.grid(True, alpha=0.5, axis='y')
fig.savefig("News-VLC-line.png", dpi=800)
plt.show()