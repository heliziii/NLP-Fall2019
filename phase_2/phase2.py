import pandas as pd
import string
import nltk
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn import preprocessing
import numpy as np
from scipy import spatial
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split




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
    len_doc = len(file['Title'])
    file['Title'] = file['Title'].str.lower()
    file['Text'] = file['Text'].str.lower()
    file['Title'] = file['Title'].apply(remove_punctuations)
    file['Text'] = file['Text'].apply(remove_punctuations)

    #remove numbers
    file['Title'] = file['Title'].str.replace("\d+","")
    file['Text'] = file['Text'].str.replace("\d+","")
    return file

def word_dic(file):
    file['all'] = file['Title'] + str(" ") + file['Text']
    words = " ".join(file['all'])
    words = word_tokenize(words)
    words = [i for i in words if not i in stop_words]
    lemm_words = []
    for ww in words:
        lemm_words.append(lemma.lemmatize(ww))
    return list(sorted(set(lemm_words)))



english = read_file("phase2_train.csv")

english = preprocess(english)
dic = word_dic(english)


print(dic)


def positional_index(file):
    index = {x: {} for x in dic}

    for i in range(len(file)):
        list_of_words = word_tokenize(file['all'].iloc[i])
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
print("finish indexing")

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




english_test = read_file("phase2_test.csv")
english_test = preprocess(english_test)
english_test['all'] = english_test['Title'] + str(" ") + english_test['Text']

##KNN


def knn_validate(k):
    print("hi")
    train = english.sample(5000)

    X_train, X_test, y_train, y_test = train_test_split(train['all'], train['Tag'], test_size = 0.1, random_state = 42)
    y_train = np.array(list(y_train))
    train_index = positional_index(X_train)
    w_t_d_train = wtd(X_train, train_index)
    test_index = positional_index(X_test)
    wtd_test = wtd(X_test, test_index)
    best_all = [0 for i in range(len(X_test))]
    for i in range(len(X_test)):
        print("i = ",i)
        doc = wtd_test[i]
        scores = [0 for i in range(len(X_train))]
        for m in range(len(w_t_d_train)):
            train_doc = w_t_d_train[m]
            scores[m] = (1 - spatial.distance.cosine(doc, train_doc))
        indexes = np.array(scores).argsort()[-k:][::-1]

        tags = y_train[indexes]
        print(tags)
        best_all[i] = max(set(list(tags)), key=list(tags).count)

    return  metrics.accuracy_score(y_test, best_all)


#print(knn_validate(9))


def knn(k):
    w_t_d_train = wtd(english, index)
    test_index = positional_index(english_test)
    wtd_test = wtd(english_test, test_index)
    best_all = [0 for i in range(len(english_test))]
    for i in range(len(english_test)):
        print(i)
        doc = wtd_test[i]
        scores = [0 for i in range(len(english))]
        for m in range(len(w_t_d_train)):
            train_doc = w_t_d_train[m]
            scores[m] = 1 - spatial.distance.cosine(doc, train_doc)
        indexes = np.array(scores).argsort()[-k:][::-1]
        tags = []
        for j in range(k):
            tags.append(english['Tag'][indexes[j]])
        best_all[i] = max(set(tags), key=tags.count)
    return best_all

# knn_prediction = knn(9)
# print(metrics.f1_score(english_test['Tag'], knn_prediction, average="micro"))
# print(metrics.accuracy_score(english_test['Tag'], knn_prediction))
# print(metrics.classification_report(english_test['Tag'],knn_prediction))


def naive_bayes():

    n_world = sum(1 for label in english['Tag'] if label == 1)
    n_sport = sum(1 for label in english['Tag'] if label == 2)
    n_business = sum(1 for label in english['Tag'] if label == 3)
    n_tech = sum(1 for label in english['Tag'] if label == 4)

    T_ct = [[0, 0, 0, 0] for i in range(len(dic))]

    def number_of_words(file):
        words = " ".join(file['all'])
        words = word_tokenize(words)
        words = [i for i in words if not i in stop_words]
        return len(words)

    n_token_world = number_of_words(english[english['Tag'] == 1])
    n_token_sport = number_of_words(english[english['Tag'] == 2])
    n_token_business = number_of_words(english[english['Tag'] == 3])
    n_token_tech = number_of_words(english[english['Tag'] == 4])

    for i in range(len(dic)):
        word = dic[i]
        lst = index[word]
        for doc_id in lst.keys():
            T_ct[i][english['Tag'][doc_id] - 1] += len(index[word][doc_id])

    for i in range(len(dic)):
        T_ct[i][0] = (T_ct[i][0] + 1) / (n_token_world + len(dic))
        T_ct[i][1] = (T_ct[i][1] + 1) / (n_token_sport + len(dic))
        T_ct[i][2] = (T_ct[i][2] + 1) / (n_token_business + len(dic))
        T_ct[i][3] = (T_ct[i][3] + 1) / (n_token_tech + len(dic))

    best_all = [0 for i in range(len(english_test))]
    for i in range(len(english_test)):
        world_score = math.log(n_world)
        sport_score = math.log(n_sport)
        business_score = math.log(n_business)
        tech_score = math.log(n_tech)
        list_of_words = word_tokenize(english_test['all'].iloc[i])
        for j in range(len(list_of_words)):
            word = list_of_words[j]
            if word not in stop_words:
                stem_word = lemma.lemmatize(word)
                if stem_word not in dic:
                    continue
                world_score += math.log(T_ct[dic.index(stem_word)][0])
                sport_score += math.log(T_ct[dic.index(stem_word)][1])
                business_score += math.log(T_ct[dic.index(stem_word)][2])
                tech_score += math.log(T_ct[dic.index(stem_word)][3])
        best_all[i] = np.argmax([world_score,sport_score,business_score,tech_score]) + 1
    return best_all

#naive_prediction = naive_bayes()






def random_forest():
    w_t_d_train = wtd(english, index)
    test_index = positional_index(english_test)
    wtd_test = wtd(english_test, test_index)
    rf = RandomForestClassifier(max_depth=10)
    rf.fit(w_t_d_train, english['Tag'])
    prediction = rf.predict(wtd_test)
    return prediction

# random_forest_prediction = random_forest()
# print(random_forest_prediction[0:50])
#
# print(metrics.f1_score(english_test['Tag'], random_forest_prediction, average="micro"))
# print(metrics.accuracy_score(english_test['Tag'], random_forest_prediction))
# print(metrics.classification_report(english_test['Tag'],random_forest_prediction))


def svm():
    w_t_d_train = wtd(english, index)
    test_index = positional_index(english_test)
    wtd_test = wtd(english_test, test_index)
    clf = SVC(kernel='linear', C=0.5)
    clf.fit(w_t_d_train, english['Tag'])
    prediction = clf.predict(wtd_test)
    return prediction


# svm_prediction = svm()
# print(svm_prediction[0:50])
#
# print(metrics.f1_score(english_test['Tag'], svm_prediction, average="micro"))
# print(metrics.accuracy_score(english_test['Tag'], svm_prediction))
# print(metrics.classification_report(english_test['Tag'],svm_prediction))



def naive_bayes_phase1(test):

    n_world = sum(1 for label in english['Tag'] if label == 1)
    n_sport = sum(1 for label in english['Tag'] if label == 2)
    n_business = sum(1 for label in english['Tag'] if label == 3)
    n_tech = sum(1 for label in english['Tag'] if label == 4)

    T_ct = [[0, 0, 0, 0] for i in range(len(dic))]

    def number_of_words(file):
        words = " ".join(file['all'])
        words = word_tokenize(words)
        words = [i for i in words if not i in stop_words]
        return len(words)

    n_token_world = number_of_words(english[english['Tag'] == 1])
    n_token_sport = number_of_words(english[english['Tag'] == 2])
    n_token_business = number_of_words(english[english['Tag'] == 3])
    n_token_tech = number_of_words(english[english['Tag'] == 4])

    for i in range(len(dic)):
        word = dic[i]
        lst = index[word]
        for doc_id in lst.keys():
            T_ct[i][english['Tag'][doc_id] - 1] += len(index[word][doc_id])

    for i in range(len(dic)):
        T_ct[i][0] = (T_ct[i][0] + 1) / (n_token_world + len(dic))
        T_ct[i][1] = (T_ct[i][1] + 1) / (n_token_sport + len(dic))
        T_ct[i][2] = (T_ct[i][2] + 1) / (n_token_business + len(dic))
        T_ct[i][3] = (T_ct[i][3] + 1) / (n_token_tech + len(dic))

    best_all = [0 for i in range(len(test))]
    for i in range(len(test)):
        world_score = math.log(n_world)
        sport_score = math.log(n_sport)
        business_score = math.log(n_business)
        tech_score = math.log(n_tech)
        list_of_words = word_tokenize(test['all'].iloc[i])
        for j in range(len(list_of_words)):
            word = list_of_words[j]
            if word not in stop_words:
                stem_word = lemma.lemmatize(word)
                if stem_word not in dic:
                    continue
                world_score += math.log(T_ct[dic.index(stem_word)][0])
                sport_score += math.log(T_ct[dic.index(stem_word)][1])
                business_score += math.log(T_ct[dic.index(stem_word)][2])
                tech_score += math.log(T_ct[dic.index(stem_word)][3])
        best_all[i] = np.argmax([world_score,sport_score,business_score,tech_score]) + 1
    return best_all

