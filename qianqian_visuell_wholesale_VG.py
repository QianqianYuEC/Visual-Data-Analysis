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



def run_pca_dbscan(features: pd.DataFrame, eps:int, min_samples:int )-> None:
    pca = PCA(n_components=2)
    pca_trans = pca.fit_transform(features)
    #print(pca_trans.shape)
    dbscan = DBSCAN(eps=eps,min_samples=min_samples)
    clustering = dbscan.fit(features)
    labels = clustering.labels_
    labels = pd.DataFrame(labels)

    #pairplot efter pca_dbscan
    """
    pca_trans = pd.DataFrame(pca_trans)
    pca_trans["labels"] = labels
    sns.pairplot(pca_trans, hue = 'labels')
    plt.title("Pairplot of wholwsale data efter pca and dbscan")
    plt.show()
    
    """
    plt.scatter(x=pca_trans[:,0], y=pca_trans[:,1], c= labels)
    #plt.colorbar(boundaries=np.arange(6)-0.5).set_ticks(np.arange(5))
    plt.title('Scatterplot of the 2-dimensional data with pca_dbscan')
    plt.show()
   

def run_umap(features:pd.DataFrame)-> None:
    umap = UMAP(random_state=0)
    umap_trans = umap.fit_transform(features)
    print(umap_trans.shape)
    plt.scatter(umap_trans[:,0], umap_trans[:,1])
    plt.title('Scatterplot of the 2-dimensional data with UMAP')
    plt.show()


def pca_plot_explained_variance(features:pd.DataFrame, plot_range: int, sum_range: int = 6) -> None:
    pca = PCA(n_components=plot_range) #?if not n_components=plot_range, always get 99.99%
    pca_full = pca.fit(features)
    print(f'Sum of the {plot_range} most important features:{sum(pca.explained_variance_ratio_[:sum_range])}')
    plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
    plt.xlabel('# of components')
    plt.ylabel('Cumulative explained variance')
    plt.title("Amount of total variance included in the principal components")
    plt.show()


def main():

    data_path_wholesale = 'C:/Users/QianqianYu/OneDrive/ECstudy/qianqian_exercise/Qianaian_Visuell_dataanalys_VG/Wholesale_customers_data.csv'
    data_wholesale = pd.read_csv(data_path_wholesale)
    print(data_wholesale.shape)
    plot = False
    if plot:
        sns.pairplot(data=data_wholesale)
        plt.title('Pairplot of wholwsale data')
        plt.show()

    #Exclude two categrate features
    data_wholesale.drop(['Channel','Region'],axis=1,inplace=True)
    print(data_wholesale.shape)
    
    #Normalisering
    data_wholesale = (data_wholesale-data_wholesale.mean())/data_wholesale.std()
    
    
    pca_plot_explained_variance(data_wholesale, plot_range = 4)
    run_umap(data_wholesale)
    run_pca_dbscan(data_wholesale,0.5,3)
    


if __name__ == "__main__":
    main()

