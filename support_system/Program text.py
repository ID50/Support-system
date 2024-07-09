'''
data_N.csv - original data
data_Nor.csv - normalized original data
data_class.csv - normalized data (a column indicating the “workload” cluster has been added)
k#_class.csv - normalized data falling into # the "workload" cluster, indicating outliers (-1)
N_k#_class.csv- original data falling into # the "workload" cluster, indicating outliers (-1)
LL_UL_#.csv - ile with the lower and upper limits of the normative intervals of the “workload” cluster of indicators of publication activity:
                1 line - lower bound for normalized data
                Line 2 - upper limit for normalized data
                3 line - lower limit for the original data
                4 line - upper limit for the original data
ST_k#_class.csv - the same as k#_class.csv, but the data is standardized along the boundaries of normative intervals (outliers are classified (F))
'''
import matplotlib.pyplot as plt
# import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import pandas as pd

# read the source data from the file
filedata = pd.read_csv('1/data_N.csv', sep=';')

for p in filedata.columns:
    filedata[p] = filedata[p].fillna(0)  # filling in the gaps in the column of the source table
    # k = filedata[p].isna().sum()  # identifying missing values ​​in columns
    # print(p, k, filedata[p].dtype)

# Calculation of the total methodological load (M) taking into account development complexity coefficients
filedata['M'] = [0] * len(filedata['id'])
for p, k in [('not_ISBN', 1), ('ISBN', 2), ('OC', 5)]:
    filedata['M'] = filedata['M'] + filedata[p] * k

# # Calculation of the total “weight of co-authorship”
# filedata['W'] = [0] * len(filedata['id'])
# for p in ['wSW', 'wR']:
#     filedata['W'] = filedata['W'] + filedata[p]

# normalize the source data
for p in filedata.columns:
    if p != 'id':
        if filedata[p].max() - filedata[p].min()!=0:
            filedata[p] = (filedata[p] - filedata[p].min()) / (filedata[p].max() - filedata[p].min())
        else:
            filedata[p] =0


# write to file
filedata.to_csv('1/data_Nor.csv', sep=';', index=False)

# We reduce the significance of the “position” indicator
filedata.d = filedata.d / 10


tabl = filedata[['s', 't','M','d','wSW', 'wR']]  # 'd','wSW', 'wR','kSW', 'SW', 'kR', 'R','ds','ISBN', 'not_ISBN', 'OC',

# reduce the dimension for visualization

from sklearn.manifold import TSNE

#selection of perplexity (effective number of “neighbors” for points)
for pp in [45]:#[30, 31, 33, 43, 44, 45, 49]:  # range(5,51):
    # pp = int(len(tabl['d']) ** 0.5)
    tsne = TSNE(n_components=2, perplexity=pp)
    filedata_2d = tsne.fit_transform(tabl)

    tsne = TSNE(n_components=3, perplexity=pp)
    filedata_3d = tsne.fit_transform(tabl)

    # separate TNSE 2d visualization
    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("t-SNE 2D Perp=" + str(pp), fontsize=16)

    plt.scatter(filedata_2d[:, 0], filedata_2d[:, 1], s=200, alpha=0.5)
    plt.grid(color='grey',
             linestyle='-.', linewidth=0.3,
             alpha=0.2)
    plt.show()
    # # separate TNSE 3d visualization
    # fig = plt.figure(figsize=(5, 5))
    # ax = fig.add_subplot(111, projection='3d')
    # plt.title("t-SNE 3D Perp=" + str(pp), fontsize=16)
    # ax.scatter(filedata_3d[:, 0], filedata_3d[:, 1], filedata_3d[:, 2], s=200, alpha=0.5)
    # ax.grid(color='grey',
    #         linestyle='-.', linewidth=0.3,
    #         alpha=0.2)
    # plt.show()



# cluster (cut out the necessary columns) id,d,s,ds,t,wSW,wR,kSW,SW,kR,R,ISBN,not_ISBN,OC,kPt,Pt,wPt
tabl = filedata[['s', 't','M','d']]
# cluster
'''
# DBSCAN
e=0.2
samples=5#int(len(filedata['ID'])**0.5)
# print(samples)
dbscan=DBSCAN(eps=e,min_samples=samples)#, metric='cosine')
clusters=dbscan.fit_predict(tabl)

'''
# Kmeans
k = 3
k_means = KMeans(n_clusters=k)
clusters = k_means.fit_predict(tabl)


print('Clusters:')
print(clusters)

# adding a column with clusters "workload" to the normalized dataframe filedata
filedata['claster'] = clusters
# put the dataframe into a file data_class.csv  *******
filedata.to_csv('1/data_class.csv', index=False, sep=';')

# visualization of three selected parameters taking into account the color of the “workload” cluster
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
# ss=100+tabl['d']*100 #can be used for dynamic marker size s=ss plasma cividis
sctt = ax.scatter(filedata['M'], filedata['t'], filedata['s'], c=filedata['claster'], cmap='cividis', s=150, marker='o',
                  alpha=0.5)  # ,label='ППС')
ax.grid(color='grey',
        linestyle='-.', linewidth=0.3,
        alpha=0.2)
ax.set_xlabel('Methodical workload', fontsize=16)
ax.set_ylabel('Auditory work', fontsize=16)
ax.set_zlabel('Average daily earnings', fontsize=16)
ax.set_title("K-means method", fontsize=16)
fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=10, ticks=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8], label='цвет номера кластера')
# ax.legend() #if you need a legend you need to add label='....'
plt.show()

# # *******************************************************************************************************************
#
# read the original data (unnormalized) from the file
N_filedata = pd.read_csv('1/data_N.csv', sep=';')
# add a column with the cluster number to the dataframe
N_filedata['claster'] = clusters


# filter the data according to their cluster number
print('Available clusters: ', filedata.claster.unique())
for cl in filedata.claster.unique():
    N_fd_k0 = N_filedata[N_filedata["claster"] == cl]  # ненормированные данные
    fd_k0 = filedata[filedata["claster"] == cl]  # нормированные данные отдельного кластера "загруженности"
    print('filter the data according to their cluster numberв')
    print('Clusters ' + str(cl))
    print(fd_k0.head())
    #cluster (cut out the necessary columns) id	d	s	ds	t	wSW	wR	kSW	SW	kR	R	ISBN	not_ISBN	OC	kPt	Pt	wPt	M
    N_tabl = N_fd_k0[
        ['wSW', 'wR', 'wPt', 'SW', 'R', 'Pt','kSW','kR']]
    tabl = fd_k0[
        ['wSW', 'wR', 'wPt', 'SW', 'R', 'Pt','kSW','kR']]

    # We standardize the indicators (according to the obtained values ​​of standard intervals)
    LL = {}  # bounds for normalized data
    UL = {}
    N_LL = {}  # bounds for unnormalized data
    N_UL = {}
    LL_UL_tabl = pd.DataFrame()
    for p in tabl:
        IQR = tabl[p].quantile(0.75) - tabl[p].quantile(0.25)
        Me = tabl[p].median()
        LL[p] = Me - 1.5 * IQR
        UL[p] = Me + 1.5 * IQR
        N_IQR = N_tabl[p].quantile(0.75) - N_tabl[p].quantile(0.25)
        N_Me = N_tabl[p].median()
        N_LL[p] = N_Me - 1.5 * N_IQR
        N_UL[p] = N_Me + 1.5 * N_IQR

    LL_UL_tabl=pd.concat([LL_UL_tabl,pd.DataFrame([LL])])
    LL_UL_tabl = pd.concat([LL_UL_tabl, pd.DataFrame([UL])])
    LL_UL_tabl = pd.concat([LL_UL_tabl, pd.DataFrame([N_LL])])
    LL_UL_tabl = pd.concat([LL_UL_tabl, pd.DataFrame([N_UL])])
    LL_UL_tabl.reset_index()
    print(LL_UL_tabl.head())

    # write the boundaries to a file d:/1/LL_UL_#.csv ******************************
    if cl == -1:
        cll = 'v'
    else:
        cll = str(cl)
    namefile = '1/LL_UL_' + cll + '.csv'
    LL_UL_tabl.to_csv(namefile, index=False, sep=';')


    # we standardize normalized indicators according to the boundaries of normative intervals
    def standart(x, LL, UL):
        if x < LL:
            return -1
        elif x <= UL:
            return 0
        else:
            return x


    ST_tabl = fd_k0[['wSW', 'wR', 'wPt', 'SW', 'R', 'Pt','kSW','kR']]
    for p in tabl:
        ST_tabl[p] = ST_tabl[p].apply(lambda x: standart(x, LL[p], UL[p]))

    # DBSCAN looking for “outliers” in standardized data
    e = 0.5
    samples = int(len(fd_k0['d']) ** 0.5)
    dbscan = DBSCAN(eps=e, min_samples=samples)
    clusters = dbscan.fit_predict(ST_tabl)

    '''
    # Kmeans
    k = 2
    k_means = KMeans(n_clusters=k)
    clusters = k_means.fit_predict(ST_tabl)
    '''

    # add a column with the cluster number to the dataframe of standardized data
    ST_tabl['claster'] = clusters


    # Preview
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    ss = fd_k0['d'] * 200  # can be used for dynamic marker size s=ss

    # visualization with feature dimension reduction
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    filedata_pca_3d = pca.fit_transform(ST_tabl)
    sctt = ax.scatter(filedata_pca_3d[:, 0], filedata_pca_3d[:, 1], filedata_pca_3d[:, 2], c=ST_tabl['claster'],
                      cmap='cividis', s=200,
                      marker='o', alpha=0.5)  # ,label='PPC')


    fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=10, ticks=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                 label='Сluster color number')

    ax.grid(color='grey',
            linestyle='-.', linewidth=0.3,
            alpha=0.2)
    ax.set_title("Cluster № " + str(cl), fontsize=16)

    plt.show()

    # add a column with the cluster number to the dataframe
    fd_k0['claster'] = clusters  # normalized data
    N_fd_k0['claster'] = clusters  # unnormalized data
    print('Selected Clusters:')
    print(fd_k0.claster.unique())
    # put the dataframe into a file d:/k#_class.csv ******************************
    if cl == -1:
        cll = 'v'
    else:
        cll = str(cl)
    namefile = '1/k' + cll + '_class.csv'
    fd_k0.to_csv(namefile, index=False, sep=';')

    # put the dataframe into a file d:/N_k#_class.csv ******************************
    if cl == -1:
        cll = 'v'
    else:
        cll = str(cl)
    namefile = '1/N_k' + cll + '_class.csv'
    N_fd_k0.to_csv(namefile, index=False, sep=';')

    #We calculate the final indicator of publication activity for each employee of the current cluster (SumID)
    ST_tabl['SumID']=ST_tabl['wSW']+ ST_tabl['wR']+ST_tabl[ 'kSW']+ST_tabl[ 'SW']+ST_tabl[ 'kR']+ST_tabl[ 'R']
    #Let’s calculate the maximum and minimum values of the final publication activity of typical cluster representatives (SumIDmax,SumIDmin)
    SumIDmax=ST_tabl.loc[ST_tabl['claster']!=-1]['SumID'].max()
    SumIDmin = ST_tabl.loc[ST_tabl['claster'] != -1]['SumID'].min()
    print('SumIDmin=',SumIDmin,'SumIDmax=',SumIDmax)

    #classify "outlier"
    def clas(x, SumIDmin, SumIDmax):
        if x < SumIDmin:
            return -1
        elif x <=SumIDmax:
            return 0
        else:
            return 1

    ST_tabl['F']=ST_tabl['SumID'].apply(lambda x: clas(x, SumIDmin, SumIDmax))

    #add ID value
    ST_tabl['id'] = fd_k0['id']

    # put part of the standardized dataframe into a file d:ST_k#_class.csv ******************************
    if cl == -1:
        cll = 'v'
    else:
        cll = str(cl)
    namefile = '1/ST_k' + cll + '_class.csv'
    ST_tabl.to_csv(namefile, index=False, sep=';')

    #  visualization of "emission" classification
    V=ST_tabl.loc[ST_tabl['claster']==-1]#we select only “outliers” from the data
    fig = plt.figure(figsize=(12, 8))
    plt.subplot(1, 1, 1)
    plt.title('"Workload" cluster №'+str(cll), fontsize=16)
    # plt.xlabel('IDs of employees who fall into the "outlier" category',fontsize=16)
    plt.ylabel('SumID',fontsize=16)
    plt.scatter(V['id'], V['SumID'], s=200, alpha=0.5)
    plt.grid(color='grey',
             linestyle='-.', linewidth=0.3,
             alpha=0.2)
    plt.plot(V['id'], [SumIDmin] * len(V['id']), color='g', linestyle='dashed',
             label='Minimum SumID of typical cluster representatives')
    plt.plot(V['id'], [SumIDmax] * len(V['id']), color='r', label='Maximum SumID of typical cluster representatives')

    # number the points
    for i,row in V.iterrows():
        plt.annotate('ID'+str(int(row['id'])), (row['id']-1,row['SumID']+0.05),fontsize=12)
    if cl==1:
        plt.annotate('Minimum SumID among typical cluster representatives', xy=(10, SumIDmin+0.02),fontsize=16, xycoords='data',
                     xytext=(1 / 5, 1 / 5), textcoords='figure fraction',
                     arrowprops=dict(facecolor='g'))
        plt.annotate('Maximum SumID among typical cluster representatives', xy=(10, SumIDmax+0.02),fontsize=16, xycoords='data',
                     xytext=(1 / 5, 3 * 1 / 4), textcoords='figure fraction',
                     arrowprops=dict(facecolor='r'))

    plt.show()

