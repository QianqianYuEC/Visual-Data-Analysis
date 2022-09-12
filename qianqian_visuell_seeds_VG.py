import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, k_means
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.decomposition import PCA



def kmeans_seeds(features: pd.DataFrame, labels:pd.DataFrame,n_clusters:int)-> None:
    kmeans = KMeans(n_clusters = n_clusters)
    kmeans.fit(features)
    kmeans_trans = kmeans.transform(features)
    print(kmeans_trans.shape)
    sns.scatterplot(x=kmeans_trans[:,0], y=kmeans_trans[:,1], hue=labels, palette=sns.hls_palette(n_clusters), legend='full')
    plt.title('Scatterplot of the 2-dimensional data with kmeans')
    plt.show()


def run_tsne(features:pd.DataFrame, labels:pd.DataFrame,n_clusters:int)-> None:
    features = np.asarray(features)
    labels = np.asarray(labels)
    tsne = TSNE(random_state=0, perplexity=5)
    tsne_trans = tsne.fit_transform(features)
    sns.scatterplot(x=tsne_trans[:,0], y=tsne_trans[:,1], hue=labels, palette=sns.hls_palette(n_clusters), legend='full')
    plt.title('Scatterplot of the 2-dimensional data with TSNE')
    plt.show()


def run_pca(features:pd.DataFrame,labels:pd.DataFrame, n_components:int) -> None:
    pca = PCA(n_components = n_components)
    pca_trans = pca.fit_transform(features)
    plt.scatter(pca_trans[:,0], pca_trans[:,1], c = labels)
    plt.colorbar(boundaries=np.arange(6)-0.5).set_ticks(np.arange(5))
    plt.title('Scatterplot of the 2-dimensional data with PCA')
    plt.show()


def main():

    data_path_seeds = 'C:/Users/QianqianYu/OneDrive/ECstudy/qianqian_exercise/Qianaian_Visuell_dataanalys_VG/seeds.csv'
    data_seeds = pd.read_csv(data_path_seeds)
    print(data_seeds.shape)
    sns.pairplot(data_seeds,hue = "Type")
    plt.show()
    labels_seeds = data_seeds.pop("Type")
    #Normalisering
    data_seeds = (data_seeds-data_seeds.mean())/data_seeds.std()
    data_seeds["Type"] = labels_seeds
    sns.pairplot(data_seeds,hue ="Type")
    plt.show()
    
    data_seeds.pop("Type")

    run_pca(data_seeds,labels_seeds,2)
    run_tsne(data_seeds,labels_seeds,3)
    kmeans_seeds(data_seeds,labels_seeds,3)

if __name__ == "__main__":
    main()

