import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.datasets import load_digits
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from sklearn.preprocessing import scale
from sklearn import metrics
from matplotlib import cm


from time import time

from tqdm import tqdm


digits = load_digits()

train_df = digits.data
label = digits.target


train_dfs = scale(digits.data)

#train_df_norm = normalize(train_df, norm='l1')
#train_df_norm2 = normalize(train_df, norm='l2')

#train_df = train_df_norm2


#######################################################################
#Showing images
#plt.gray()
#plt.matshow(digits.images[0]) 
#plt.show() 
#
#plt.imshow(digits.images[3])
#
#for i in range(0,9):
#    plt.imsave('{}_img.png'.format(i),digits.images[i],dpi=600)

#Question 1
#Apply K-Means to your dataset. Use adjusted rand index to evaluate your results. 
#Run K-Means for multiple rounds with different random initializations. Do the results vary for different initializations?  
#######################################################################
wcss = []
for i in tqdm(range(1, 30)):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 47)
    kmeans.fit(train_df)
    wcss.append(kmeans.inertia_)
wcss=np.array(wcss)
#np.save('wcss_norm.npy',wcss)
    


fig = plt.figure()
ax = plt.axes()


#Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 30), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.grid()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1E'))
#plt.savefig('clusters_vs_wcss.png',bbox_inches="tight",dpi=600)
#####################################################################


#####################################################################

ari=[]
for i in tqdm(range(1,31)):
    kmeans = KMeans(n_clusters = i, init = 'random',  random_state = 47)
    y_kmeans = kmeans.fit_predict(train_df)
    ari.append(adjusted_rand_score(label,y_kmeans))

ari.insert(0,0)    

plt.plot(ari)
plt.xlim([1,31])
plt.title('ARI of Various Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Adjusted Rand Index')
plt.grid()
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], fontsize=8)
#plt.savefig('ARI_various_clusters.png',dpi=600)

    
#At 10 clusters
kmeans = KMeans(n_clusters = 10, init = 'random',random_state=47)
y_kmeans = kmeans.fit_predict(train_df)




adjusted_rand_score(label,y_kmeans)

#Various random states, yes they do vary for different initializations
ari_rand=[]
centroid_rand=[]
for i in tqdm(range(0,31)):
    kmeans = KMeans(n_clusters = 10, init = 'random')
    y_kmeans = kmeans.fit(train_df)
    centroid_rand.append(kmeans.cluster_centers_)
    new_labels = kmeans.labels_
    ari_rand.append(adjusted_rand_score(label,new_labels))


plt.plot(ari_rand)
plt.title('ARI of Random Initialization (10 clusters)')
plt.xlabel('Random Initialization')
plt.ylabel('Adjusted Rand Index')
plt.grid()
#plt.savefig('ARI_Rand_Init.png',dpi=600)
    

#Backup
#for i in tqdm(range(0,30)):
#    kmeans = KMeans(n_clusters = 10, init = 'random')
#    y_kmeans = kmeans.fit_predict(train_df)
#    ari_rand.append(adjusted_rand_score(label,y_kmeans))
    

#Question2
#Evaluate how results change with respect to parameter K (using adjusted rand index). 
#Since random initialization impacts the results, run K-means multiple times and show the average result
##################################################################################################################

#Different Ks with random init
ari_diffK = []
for i in tqdm(range(1,31)):
    kmeans = KMeans(n_clusters = i, init='random')
    y_kmeans = kmeans.fit_predict(train_df)
    ari_diffK.append(adjusted_rand_score(label,y_kmeans)) 
    
avg_ari_diffK = sum(ari_diffK)/len(ari_diffK)

ari_diffK.insert(0,0)

plt.plot(ari_diffK)
plt.xlim([1,31])
plt.title('ARI of Different K - Random Initialization')
plt.xlabel('Clusters')
plt.ylabel('Adjusted Rand Index')
plt.grid()
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], fontsize=8)
#plt.savefig('Different_K_Rand_Init.png', dpi=600)



#Question 3
#Apply Gaussian mixture model and evaluate the results (using adjusted rand index). Try several different K.
##############################################################################################################
#Gaussian Mixed Models

ari_gmm=[]
for i in tqdm(range(1,31)):
    gmm = GaussianMixture(n_components=i, random_state=47)
    gmm.fit(train_df)
    y_cluster_gmm = gmm.predict(train_df)
    ari_gmm.append(adjusted_rand_score(label,y_cluster_gmm))

ari_gmm.insert(0,0)

plt.plot(ari_gmm)
plt.xlim([1,31])
plt.title('ARI of GMM')
plt.xlabel('Clusters')
plt.ylabel('Adjusted Rand Index')
plt.grid()
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], fontsize=8)
#plt.savefig('ARI_various_clusters_GMM.png', dpi=600)
    
#GMM at 10 cluster
gmm = GaussianMixture(n_components=10, random_state=47)
gmm.fit(train_df)
y_cluster_gmm = gmm.predict(train_df)
adjusted_rand_score(label,y_cluster_gmm)
    
    
#Question 4
#Use PCA to reduce the feature space to 2 dimensions. Apply K-Means and Gaussian mixture to 2-dimensional data. 
#Compare the results using scatter plot. Try several different numbers of clusters. 
#############################################################################################################

#PCA
pca = PCA(n_components=64, random_state=45)
X_pca_array = pca.fit_transform(train_df)
#pca = PCA(n_components=10).fit(train_df)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance') # we see about 30 first pc explains ~90% of the variance of the data
plt.title('PCA Components vs Explained Variance')
#plt.savefig('pca_explained.png',dpi=600)

#We use 2 PCs as instructed
pca = PCA(n_components=2, random_state=45)
X_pca_array = pca.fit_transform(train_df)

train_df_pd = pd.DataFrame(train_df)
label_pd = pd.DataFrame(label)

label_pd.columns = ['label']
groups = label_pd.groupby('label')



#PCA components =30
pca = PCA(n_components=30, random_state=45)
X_pca_array = pca.fit_transform(train_df)
ari_kmeans_pca30=[]
kmeans_pca = KMeans(n_clusters = 10 , init='random', random_state=45)
y_pred_kmeans_pca30 = kmeans_pca.fit_predict(X_pca_array)
adjusted_rand_score(label,y_pred_kmeans_pca30)
label_pred = pd.DataFrame(y_pred_kmeans_pca30)
label_pred.columns = ['label']


plt.scatter(X_pca_array[:,0],X_pca_array[:,1],c=label_pred['label'].map(cm.Paired_r))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Label Predictions for 30 PCs')
#plt.savefig('label_predictions_class.png', dpi=400)


colors = {0:'zero',1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven',8:'eight',9:'nine'}

plt.scatter(X_pca_array[:,0],X_pca_array[:,1],c=label_pd['label'].map(cm.Paired_r))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Original Label for 30 PCs')
#plt.savefig('original_label_class.png', dpi=400)

plt.legend()


plt.legend()



#Try K_means on PCA-ed data
ari_kmeans_pca=[]
for i in tqdm(range(1,31)):
    kmeans_pca = KMeans(n_clusters = i , init='random', random_state=45)
    y_pred_kmeans_pca = kmeans_pca.fit_predict(X_pca_array)
    ari_kmeans_pca.append(adjusted_rand_score(label,y_pred_kmeans_pca))



    
    

   
ari_kmeans_pca.insert(0,0)    
    
plt.plot(ari_kmeans_pca)
plt.xlim([1,31])
plt.title('ARI of Kmeans (PCA)')
plt.xlabel('Clusters')
plt.ylabel('Adjusted Rand Index')
plt.grid()
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], fontsize=8)
#plt.savefig('ARI_Kmeans_PCA.png',dpi=600)


#X_pca = pd.DataFrame(X_pca_array, columns=['PC1','PC2']) 
#plt.scatter(X_pca['PC1'],X_pca['PC2'], s=10, c=y_pred_kmeans_pca) 
#plt.xlabel('PC1')
#plt.ylabel('PC2')
#plt.savefig('PCA.png',dpi=600)


#Try GMM on PCA-ed data
ari_gmm_pca=[]
for i in tqdm(range(1,31)):
    gmm = GaussianMixture(n_components=i, random_state=45)
    gmm.fit(X_pca_array)
    y_cluster_gmm = gmm.predict(X_pca_array)
    ari_gmm_pca.append(adjusted_rand_score(label,y_cluster_gmm))
    
ari_gmm_pca.insert(0,0)    
    
plt.plot(ari_gmm_pca)
plt.xlim([1,31])
plt.title('ARI of GMM (PCA)')
plt.xlabel('Clusters')
plt.ylabel('Adjusted Rand Index')
plt.grid()
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], fontsize=8)
#plt.savefig('ARI_GMM_PCA.png',dpi=600)




