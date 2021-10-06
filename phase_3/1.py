import pandas as pd
import string
import nltk
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn import preprocessing
import numpy as np
import gensim
from sklearn.cluster import  KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import pylab as pl
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score



# def find_optimal_clusters(vectors, max_k):
#     iters = range(10, max_k + 1, 2)
#
#     sse = []
#     # for k in iters:
#         sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(vectors).inertia_)
#         print('Fit {} clusters'.format(k))
#     print(sse)
#find_optimal_clusters(wtd_vectors, 40)


def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation,' ')
    return text

lemma = nltk.wordnet.WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words = [remove_punctuations(x) for x in stop_words]
stop_words.append('cannot')


def read_file(address):
    return  pd.read_csv(address)



def preprocess(file):
    len_doc = len(file['ID'])
    file['Text'] = file['Text'].str.lower()
    file['Text'] = file['Text'].apply(remove_punctuations)

    #remove numbers
    file['Text'] = file['Text'].str.replace("\d+"," ")
    file['Text'] = file['Text'].str.replace("[^a-zA-Z]", " ")
    return file

def word_dic(file):
    words = " ".join(file['Text'])
    words = word_tokenize(words)
    words = [i for i in words if not i in stop_words]
    lemm_words = []
    for ww in words:
        lemm_words.append(lemma.lemmatize(ww))
    return list(sorted(set(lemm_words)))

def sent_word(file):
    data = []
    for k in file:
        temp = []
        for j in word_tokenize(k):
            if j not in stop_words:
                temp.append(lemma.lemmatize(j))

        data.append(temp)
    return data



english = pd.read_csv("Data.csv", encoding='latin1')

english = preprocess(english)
dic = word_dic(english)
print(dic)

def positional_index(file):
    index = {x: {} for x in dic}

    for i in range(len(file)):
        list_of_words = word_tokenize(file['Text'].iloc[i])
        for j in range(len(list_of_words)):
            word = list_of_words[j]
            if word not in stop_words:
                stem_word = lemma.lemmatize(word)
                if stem_word not in dic:
                    continue
                if bool(index[stem_word]) == False:
                    index[stem_word][i] = [j ]
                else:
                    if i in index[stem_word]:
                        index[stem_word][i].append(j)
                    else:
                        index[stem_word][i] = [j]
    return index

index = positional_index(english)

def wtd(file,index_file):
    idf_word = [0 for i in range(len(dic))]
    for i in range(len(dic)):
        word = dic[i]
        if len(index_file[word].keys()) == 0:
            idf_word[i] = 0
        else:
            idf_word[i] = math.log(len(file) / len(index_file[word].keys()))

    tf = [[0 for i in range(len(dic))] for j in range(len(file))]

    for i in range(len(dic)):
        word = dic[i]
        posting_word = index_file[word]
        for key in posting_word.keys():
            tf[key][i] = len(posting_word[key])

    w_t_d = [[0 for i in range(len(dic))] for j in range(len(file))]
    for i in range(len(file)):
        for j in range(len(dic)):
            w_t_d[i][j] = idf_word[j] * tf[i][j]

    w_t_d = preprocessing.normalize(w_t_d, norm='l2')
    return w_t_d

wtd_vectors = wtd(english['Text'],index)


data = sent_word(english['Text'])
model = gensim.models.Word2Vec(data, min_count = 1, size = 100, window = 5, sg = 1)
word_vectors = model.wv


##doc2vec
def doc_2_vec(file):
    doc2vec = []
    for j in range(len(file['Text'])):
        doc =  file['Text'][j]
        words = word_tokenize(doc)
        words = [i for i in words if not i in stop_words]
        words = [lemma.lemmatize(word) for word in words]
        if len(words) == 0:
            doc2vec.append([0 for k in range(100)])
            continue
        vecs = []
        for word in words:
            word_vec = list(word_vectors.get_vector(word))
            vecs.append(word_vec)
        doc_mean = np.array(vecs).mean(axis=0)
        doc2vec.append(doc_mean)
    return doc2vec

#doc2vec = list(doc_2_vec(english))

def find_k(file,s):
    print("finding appropriate k")
    k_range = [2,3,4,5,6]
    if s == "k":
        for i in k_range:
            km = KMeans(n_clusters=i)
            clusters = km.fit(file)
            silhouette_avg = silhouette_score(file,clusters.labels_ )
            print(silhouette_avg,i)
    if s == "g":
        for i in k_range:
            gmm = GaussianMixture(3, covariance_type='diag', random_state=0).fit(file)
            gmm_labels = gmm.predict(file)
            silhouette_avg = silhouette_score(file, gmm_labels)
            print(silhouette_avg, i)


#find_k(wtd_vectors,"k")



##kmeans
def k_means(file,s):
    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(file)

    km = KMeans(n_clusters=3)
    clusters = km.fit_predict(file)

    with open('kmeans' + s + '.txt', 'w') as f:
        for item in clusters:
            f.write("%s\n" % item)

    for i in range(0, pca_2d.shape[0]):
        if clusters[i] == 1:
            c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r', marker='+')
        elif clusters[i] == 0:
            c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g', marker='o')
        elif clusters[i] == 2:
            c3 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b', marker='*')
    pl.title('K-means clusters the word2vec dataset into 3 clusters')
    pl.show()


#gmm
def guassian_mixture(file,s):
    gmm = GaussianMixture(3, covariance_type='diag', random_state=0).fit(file)
    gmm_labels = gmm.predict(file)

    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(file)

    with open('gmm' + s + '.txt', 'w') as f:
        for item in gmm_labels:
            f.write("%s\n" % item)

    for i in range(0, pca_2d.shape[0]):
        if gmm_labels[i] == 1:
            c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r', marker='+')
        elif gmm_labels[i] == 0:
            c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g', marker='o')
        elif gmm_labels[i] == 2:
            c3 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b', marker='*')
    pl.title('GMM clusters the word2vec dataset into 3 clusters')
    pl.show()


##hierarchical
def hierarchical(file,s):
    linked = linkage(file,  method='ward', metric='euclidean')
    plt.figure(figsize=(10, 7))
    dendrogram(linked,
                truncate_mode='lastp',
                p = 10,
                orientation='top',
                distance_sort='descending',
                show_leaf_counts=True)
    plt.show()
    print("hi")
    hierarchical_clustering = AgglomerativeClustering(linkage='ward', affinity='euclidean', n_clusters=3)
    print("hi")
    clusters = hierarchical_clustering.fit_predict(file)
    with open('hierar' + s + '.txt', 'w') as f:
        for item in clusters:
            f.write("%s\n" % item)

hierarchical(wtd_vectors,"_tf-idf")