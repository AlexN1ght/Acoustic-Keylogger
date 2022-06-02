import matplotlib.pyplot as plt

import scprep
import numpy as np

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, homogeneity_score, silhouette_score


def plot_keystrokes(X, y_int, y_str, min_distance=0.05, images=None, figsize=(13, 10)):
    X_norm = MinMaxScaler().fit_transform(X)
    neighbors = np.array([[10., 10.]])

    plt.figure(figsize=figsize)

    cmap = scprep.plot.colors.tab30()
    labels = np.unique(y_int)


    for label in labels:
        plt.scatter(X_norm[y_int == label, 0], X_norm[y_int == label, 1], c=[cmap(label)])

    plt.axis('off')
    ax = plt.gcf().gca()
    for index, image_coord in enumerate(X_norm):
        closest_distance = np.linalg.norm(np.array(neighbors) - image_coord, axis=1).min()
        if closest_distance > min_distance:
            neighbors = np.r_[neighbors, [image_coord]]
            if images is None:
                if images is None:
                    plt.text(image_coord[0], image_coord[1], y_str[index],
                             color=cmap(y_int[index]+4), fontdict={"weight": "bold", "size": 16})
                else:
                    image = images[index]
                    imagebox = AnnotationBbox(OffsetImage(image, cmap="binary"), image_coord)
                    ax.add_artist(imagebox)


def show_cluster_compare(X, lbls, pred):
    tsne = TSNE(n_components=2, early_exaggeration=1000, perplexity=30, init='random', learning_rate=10,  random_state=10, n_iter=2000, metric='l2',  square_distances=True)
    X_reduced_tsne = tsne.fit_transform(X)

    fig = plt.gcf()
    fig.set_dpi(100)
    fig.set_size_inches(7, 5, forward=True)
    plt.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=lbls, cmap=scprep.plot.colors.tab40())
    plt.pause(0.05)
    fig = plt.gcf()
    fig.set_dpi(100)
    fig.set_size_inches(7, 5, forward=True)
    plt.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=pred, cmap=scprep.plot.colors.tab40())
    plt.pause(0.05)


def cluster_data(X, n_components=50, mode='GM', retries_num=1, lbls=None):
    assert mode == 'GM' or mode == 'KM'
    max_score = 0
    pred_proba = None
    print('0%', end='')
    for try_i in range(retries_num):
        
        if mode == 'GM':
            cluster_model = GaussianMixture(n_components=n_components, n_init=5, max_iter=3000, init_params='kmeans')
        else:
            cluster_model = KMeans(n_clusters=n_components, max_iter=3000, init='k-means++', algorithm="full")
        cluster_model.fit(X)
        pred = cluster_model.predict(X)
        if mode == 'GM':
            pred_proba = cluster_model.predict_proba(X)
        if lbls:
            score = homogeneity_score(lbls, pred)
        else:
            score = silhouette_score(X, pred)
        if score > max_score:
            max_score = score
            best_pred = pred
            best_pred_proba = pred_proba
        print(f'\r{(try_i + 1)/retries_num* 100:.2f}%', end='', flush=True)
    print(flush=True)
    
    if mode == 'GM':
        return best_pred
    else:
        return best_pred
    

def cluster_data_mean(X, n_components=50, mode='GM', retries_num=1, lbls=None):
    assert mode == 'GM' or mode == 'KM'
    h_s = 0
    a_r_s = 0
    print('0%', end='')
    for try_i in range(retries_num):
        
        if mode == 'GM':
            cluster_model = GaussianMixture(n_components=n_components, n_init=5, max_iter=3000, init_params='kmeans')
        else:
            cluster_model = KMeans(n_clusters=n_components, max_iter=3000, init='k-means++', algorithm="full")
        cluster_model.fit(X)
        
        pred = cluster_model.predict(X)
        h_s += homogeneity_score(lbls, pred)
        a_r_s += adjusted_rand_score(lbls, pred)
        
        print(f'\r{(try_i + 1)/retries_num* 100:.2f}%', end='', flush=True)
    print(flush=True)
    
    h_s /= retries_num
    a_r_s /= retries_num
    return a_r_s, h_s 


