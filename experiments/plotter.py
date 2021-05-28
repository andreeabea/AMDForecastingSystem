import numpy as np
import matplotlib.pyplot as plt


def plot_bar_chart(means1, means2):
    n = np.arange(len(means1))
    width = 0.35

    plt.bar(n, means1, width, label='series resampled at 1 month time intervals')
    plt.bar(n + width, means2, width,
            label='time intervals (in number of months) as inputs')

    plt.ylabel('Mean R^2 scores')
    plt.title('Visual acuity prediction from past visual acuity data')

    plt.yticks(np.arange(0, 1.3, 0.1))
    plt.xticks(n + width / 2, ('size=2', 'size=3', 'size=4'))
    plt.legend(loc='best')

    xlocs, xlabs = plt.xticks()

    for i, v in enumerate(means1):
        plt.annotate(v, xy=(xlocs[i] - 0.17, v + 0.01), ha='center')
        plt.annotate(means2[i], xy=(xlocs[i] + 0.17, means2[i] + 0.01), ha='center')
        #plt.text(xlocs[i] - 0.255, v + 0.01, str(v), ha='center')
        #plt.text(xlocs[i] + 0.05, means2[i] + 0.01, str(means2[i]))

    plt.show()


means1 = [0.9, 0.938, 0.935]
means2 = [0.818, 0.81, 0.695]
plot_bar_chart(means1, means2)
