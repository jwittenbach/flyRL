import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context('talk')
sns.set_style('whitegrid')

def value_plot(data, model):

    d = model.locs[data]

    # bar plot of local distribution over time
    edges = np.copy(model.locs)
    de = np.diff(edges)[0]
    edges = np.r_[edges[0]-de, edges] + de/2.0
    h, _ = np.histogram(d, bins=edges)
    h = 1.0*h/h.sum()
    centers = (edges + de/2.0)[:-1]
    plt.bar(centers, h, width=de, color=sns.color_palette('pastel')[0])

    # scatter plot choices over time
    m = h.max()
    plt.scatter(d, np.linspace(1.1*m, 1.1*m+1, len(d)), s=10,
            c=sns.color_palette('dark')[0])

    # line plot of value function
    x = np.linspace(model.vSpline.domain[0], model.vSpline.domain[1], 200)
    v = model.vSpline(x)
    v = m*(v - v.min())/(v.max() - v.min())
    plt.plot(x, v, c=sns.color_palette('dark')[2])

    # line plot of penalty function
    model.pFunc.p_circ = 0.5
    p = model.pFunc(x)
    inds = np.where(p<m)
    plt.plot(x[inds], p[inds], c=sns.color_palette('dark')[3])

    plt.show()
