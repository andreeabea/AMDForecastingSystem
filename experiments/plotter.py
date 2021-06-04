import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy.polynomial.polynomial import polyfit
from scipy.stats import pearsonr, spearmanr
import seaborn as sns


def plot_bar_chart(means1, means2, features_str):
    n = np.arange(len(means1))
    width = 0.35

    plt.bar(n, means1, width, label='series resampled at 1 month time intervals')
    plt.bar(n + width, means2, width,
            label='time intervals (in number of months) as inputs')

    plt.ylabel('Mean R^2 scores')
    plt.title('Visual acuity prediction from ' + features_str)

    plt.yticks(np.arange(0, 1.3, 0.1))
    plt.xticks(n + width / 2, ('1 previous visit', '2 previous visits', '3 previous visits'))
    plt.legend(loc='best')

    xlocs, xlabs = plt.xticks()

    for i, v in enumerate(means1):
        plt.annotate(v, xy=(xlocs[i] - 0.17, v + 0.01), ha='center')
        plt.annotate(means2[i], xy=(xlocs[i] + 0.17, means2[i] + 0.01), ha='center')
        #plt.text(xlocs[i] - 0.255, v + 0.01, str(v), ha='center')
        #plt.text(xlocs[i] + 0.05, means2[i] + 0.01, str(means2[i]))

    plt.show()


def plot_bar_chart_feature_sel(means1, means2, means3):
    n = np.arange(len(means1))
    width = 0.15

    plt.bar(n, means1, width, label='1 previous visit', color='#cd5c5c')
    plt.bar(n + width + 0.01, means2, width, label='2 previous visits', color='#5ccd95')
    plt.bar(n + 2 * width + 0.02, means3, width, label='3 previous visits', color='#346fa9')

    plt.ylabel('Mean cross-validated R^2 scores')
    plt.xlabel('Feature selection method')
    plt.title('Feature selection methods comparison (with resampling)')

    plt.yticks(np.arange(0, 1.3, 0.1))
    plt.xticks(n + width / 3 + width / 2, ('None', 'RFE+GBR', 'LASSO'))
    plt.legend(loc='best')

    xlocs, xlabs = plt.xticks()

    for i, v in enumerate(means1):
        plt.annotate(v, xy=(xlocs[i] - 0.13, v + 0.01), ha='center')
        plt.annotate(means2[i], xy=(xlocs[i] + 0.04, means2[i] + 0.01), ha='center')
        plt.annotate(means3[i], xy=(xlocs[i] + 0.19, means3[i] + 0.01), ha='center')
        #plt.text(xlocs[i] - 0.255, v + 0.01, str(v), ha='center')
        #plt.text(xlocs[i] + 0.05, means2[i] + 0.01, str(means2[i]))

    plt.show()


def get_vals(exp):
    if exp == 1:
        #experiment 1
        means1 = [0.937, 0.943, 0.94]
        means2 = [0.847, 0.851, 0.79]
        features_str = 'past visual acuity data'
    elif exp==2:
        #experiment 2
        means1 = [0.63, 0.827, 0.83]
        means2 = [0.55, 0.81, 0.77]
        features_str = 'all numerical OCT features'
    return means1, means2, features_str


def plot_1D_feature_importance(x, y):
    fig, ax = plt.subplots()
    fig.set_figwidth(50)
    extent = [x[0] - (x[1] - x[0]) / 2., x[-1] + (x[1] - x[0]) / 2., 0, 1]
    ax.imshow(y[np.newaxis, :], cmap="plasma", aspect="auto", extent=extent)
    ax.set_xticks(np.arange(min(x), max(x) + 1, 1.0))
    ax.set_xlim(extent[0], extent[1])

    labels = ax.get_xticklabels()
    ticks = ax.get_xticks()
    for label, tick in zip(labels, ticks):
        if y[int(tick)] > 150:
            label.set_color('white')
        else:
            label.set_fontsize(20)

    plt.tight_layout()
    plt.show()


def plot_cv(mode='resampled'):
    x = ['VA', 'Num.', 'Num.+Img', 'All']
    #x = ['1','2', '3']
    if mode == 'resampled':
        y1 = [0.96, 0.57, 0.83, 0.93]
        y2 = [0.96, 0.69, 0.88, 0.95]
        y3 = [0.96, 0.66, 0.91, 0.96]
    else:
        y1 = [0.87, 0.46, 0.50, 0.80]
        y2 = [0.88, 0.56, 0.66, 0.78]
        y3 = [0.89, 0.30, 0.60, 0.70]

    #y1 = [0.57, 0.69, 0.66]
    #y2 = [0.83, 0.88, 0.91]
    #y3 = [0.93, 0.95, 0.96]

    plt.plot(x, y1, marker="o", label="1 previous visit")
    plt.plot(x, y2, marker="o", label="2 previous visits")
    plt.plot(x, y3, marker="o", label="3 previous visits")
    plt.xlim(0, 3)
    plt.ylim(0.3, 1.2)
    plt.xlabel("Dataype used from previous visits")
    plt.ylabel("Cross-validated R^2 scores")
    plt.title("Cross-validated LSTM scores (without resampling)")
    plt.legend()
    plt.grid()
    plt.show()


def plot_predicted_vs_actual():
    testY = [0.005 ,     0.025 ,     0.025  ,    0.5 ,       0.5   ,     0.625,
     0.025 ,     0.04466667, 0.071875 ,  1.   ,      1. ,        0.93333333,
     0.8 ,       0.18373016, 0.31746032, 0.025,      0.33846154, 0.26153846,
     0.1 ,       0.40769231, 0.4      ,  0.2,        0.1,        1.,
     1. ,        0.025   ,   0.025    ,  0.025,      0.025,      0.025,
     1. ,        1.  ,       1.,         0.025,      0.31746032, 0.025,
     0.05 ,      0.625  ,    0.025 ,     0.01,       0.035,      0.045,
     0.646875 ,  0.734375  , 0.8   ,     0.025 ,     0.2,        0.075,
     0.68333333, 0.625   ,   0.25]

    testY2 = [0.005,      0.025,      0.025,      0.5,        0.5,        0.625,
 0.025,      0.04466667, 0.071875,   1.,         1.,         0.93333333,
 0.8,        0.18373016, 0.31746032, 0.025,      0.33846154, 0.26153846,
 0.1,        0.40769231, 0.4,        0.2,        0.1,        1.,
 1.,         0.025,      0.025,      0.025,      0.025,      0.025,
 1.,         1.   ,      1.,         0.025,      0.31746032, 0.025,
 0.05 ,      0.625 ,     0.025,      0.01,       0.035,      0.045,
 0.646875,   0.734375 ,  0.8,        0.025,      0.2,        0.075,
 0.68333333, 0.625 ,     0.25      ]

    x = np.arange(len(testY2))

    predicted = [0.00378278,
 0.01045772,
 0.02461845,
 0.47006026,
 0.44986805,
 0.62571514,
 0.01444137,
 0.08808237,
 0.05530041,
 0.95877486,
 0.9741603 ,
 0.94524336,
 0.66741633,
 0.18198073,
 0.19130716,
 0.01367685,
 0.2955415 ,
 0.21955961,
 0.1786952 ,
 0.39475903,
 0.45500585,
 0.18489072,
 0.0936006 ,
 0.9715599 ,
 0.9554812 ,
 0.04033974,
 0.01303664,
 0.0243105 ,
 0.01185971,
 0.02830061,
 0.98860717,
 0.98933446,
 0.99042654,
 0.04759452,
 0.31479785,
 0.03943646,
 0.04198578,
 0.52564883,
 0.0137707 ,
 0.00444365,
 0.0045006 ,
 0.01283005,
 0.5966805 ,
 0.75865495,
 0.8706368 ,
 0.00215966,
 0.04730716,
 0.04421592,
 0.6449775 ,
 0.6201006 ,
 0.2667455 ]

    predicted2 = [0.01221895,
 0.01700962,
 0.03069645,
 0.49094573,
 0.48352778,
 0.6618546 ,
 0.02010074,
 0.07768449,
 0.08006975,
 0.9778546 ,
 0.9750783 ,
 0.9444066 ,
0.80338806,
 0.18214223,
 0.16241187,
 0.02058953,
 0.33834836,
 0.26346505,
 0.14539614,
 0.387954  ,
 0.3669642 ,
 0.21609104,
 0.10645163,
 0.96571314,
 0.960547  ,
 0.04474372,
 0.01748404,
 0.02891767,
 0.02009797,
 0.03599676,
 0.98929405,
 0.98986983,
 0.9908363 ,
 0.04450938,
 0.33369103,
 0.04573259,
 0.05517647,
 0.5449236 ,
 0.02012047,
 0.01215512,
 0.01487383,
 0.02756348,
 0.55338645,
 0.72754854,
 0.8886038 ,
 0.00868914,
 0.06983852,
 0.04533359,
 0.68152565,
0.66527677,
 0.27368653]

    plt.plot(x, testY, label="actual")
    plt.plot(x, predicted, label="predicted", c='#a93434')
    plt.xlim(0, len(testY))
    plt.ylim(0.0, 1.1)
    #plt.scatter(testY, predicted)

    #sns.regplot(x=testY2, y=predicted2)

    #print(pearsonr(testY2, predicted2))
    plt.xlabel("Prediction No.")
    plt.ylabel("Visual Acuity")
    plt.title("Actual vs. predicted values for the best model")
    plt.legend()
    plt.grid()
    plt.show()


means1, means2, features_str = get_vals(2)
#plot_bar_chart(means1, means2, features_str)
# resampled
#plot_bar_chart_feature_sel([0.83, 0.73, 0.65], [0.87, 0.79, 0.69], [0.91, 0.87, 0.80])
# not resampled
#plot_bar_chart_feature_sel([0.53, 0.53, 0.51], [0.67, 0.57, 0.60], [0.70, 0.59, 0.56])
#plot_cv('no')
plot_predicted_vs_actual()

rfe_feature_ranks = [1,   1,  65,  89,  42,   7,  84,  31, 113,   1,  11, 120,  54,   1,   1,   1,   3,   1,
  99,   1,  91,   1,   1,   1, 148, 154, 121, 155,  12, 104, 126,  73,  78, 136,  41,  35,
 108, 127,  64, 183,  59,  23,   1,  45, 156,  97,   1,  94, 100, 111, 170, 219, 131,  83,
  76, 142,   1,  67, 105, 106, 145,  58, 229, 194, 210, 212,  15,  16, 165, 205, 226, 167,
  51,  29, 236, 174,  22, 161, 223,  92,  77, 209,  56,  39, 235,  88,  18,  53, 241, 218,
  43,  93, 228, 242, 252, 112, 239, 101,   1, 248,   1, 234, 246, 247,  85,  75, 151, 133,
 245, 243, 204, 214, 220, 230, 238,  90,  82,  70, 139, 153, 149,  72,   1, 130,   9, 164,
 157,  47, 109, 175, 177, 179,  14, 184, 187, 189,  40,  27, 249, 253,  79,  28,  55,  20,
  50, 202, 181, 206,   1, 200, 158,  34, 227, 203, 215, 182, 192, 141,   6, 159, 147,  86,
  30, 163,   2, 168, 173,   8, 176,  37,  10,   1, 185,   1,  52, 144,   1,  98,  17,  95,
 199 , 48, 114,  81, 191, 125, 119,  24, 198,  74, 201, 250 ,186, 217, 222,  44, 244, 232,
   1, 225, 208,  71,  87, 211, 162, 171,  68, 193, 190,  60,  33,  49,  61, 134, 196,  21,
 160, 107,  46, 166, 103, 233, 213, 231,  96, 221,   1, 240,  63,   4, 237,  36, 224,   1,
 251, 169,  69, 172, 197, 216, 188, 117, 207,   1, 195,  32,  25, 124, 128, 152, 102, 140,
  19, 178,  66,  62,   1, 180, 115, 150,  26, 135, 146, 138, 137,  13, 143,   5, 129, 116,
  38, 122,  57, 123,   1, 118, 110, 132,   1,  80]

#rfe_feature_ranks = np.array(rfe_feature_ranks)
#feature_x = [i for i in range(len(rfe_feature_ranks))]
#plot_1D_feature_importance(feature_x, rfe_feature_ranks)
