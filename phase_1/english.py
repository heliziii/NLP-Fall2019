import pandas as pd
import string
import nltk
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn import preprocessing
import numpy as np
import re


def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation,'')
    return text

lemma = nltk.wordnet.WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words = [remove_punctuations(x) for x in stop_words]
stop_words.append('cannot')


def read_file(address):
    return  pd.read_csv(address)

english = read_file("English.csv")
len_doc = len(english['Title'])

def preprocess(file):
    len_doc = len(file['Title'])
    file['Title'] = file['Title'].str.lower()
    file['Text'] = file['Text'].str.lower()
    file['Title'] = file['Title'].apply(remove_punctuations)
    file['Text'] = file['Text'].apply(remove_punctuations)

    #remove numbers
    file['Title'] = file['Title'].str.replace("\d+","")
    file['Text'] = file['Text'].str.replace("\d+","")
    file['all'] = file['Title'] + str(" ") + file['Text']
    words = " ".join(file['all'])
    words = word_tokenize(words)
    words = [i for i in words if not i in stop_words]
    lemm_words = []
    for ww in words:
        lemm_words.append(lemma.lemmatize(ww))
    cnt = Counter(lemm_words)
    most_occur = cnt.most_common(10)
    return list(sorted(set(lemm_words))), most_occur

dic, most_occur = preprocess(english)





##positional index
def positional_index(file):
    index = {x: {} for x in dic}

    for i in range(len(file['Title'])):
        list_of_words = word_tokenize(file['all'].iloc[i])
        for j in range(len(list_of_words)):
            word = list_of_words[j]
            if word not in stop_words:
                stem_word = lemma.lemmatize(word)
                if bool(index[stem_word]) == False:
                    index[stem_word][i] = [j ]
                else:
                    if i in index[stem_word]:
                        index[stem_word][i].append(j)
                    else:
                        index[stem_word][i] = [j]
    return index

index = positional_index(english)

def posting_list(query):
    if query not in dic:
        print("Word not found")
    else:
        print(index[query])


gamma_index = {x: "" for x in dic}

def gamma_code(n):
  binary_n = format(n, 'b')
  binary_offset = binary_n[1::]
  unary_length = ''.join(['1'*len(binary_offset)]) + '0'
  return unary_length + str(binary_offset)

for word in dic:
    doc_id = list(index[word].keys())
    gamma_index[word] += gamma_code(doc_id[0])
    for i in range(1,len(doc_id)):
        gap = doc_id[i] - doc_id[i-1]
        gamma_index[word] += gamma_code(gap)
print(gamma_index['abandoned'])


vlb_index = {x: "" for x in dic}

def vlb(x):
    bits = []
    while x > 0:
        bits.append(x % 2)
        x //= 2
    ret= ""
    for i in range(0, len(bits), 7):
        cur = ""
        for j in range(i, i + 7):
            if j >= len(bits):
                cur += '0'
            else:
                cur += str(bits[j])
        if i == 0:
            cur += '1'
        else:
            cur += '0'
        ret += cur
    return ret[::-1]

for word in dic:
    doc_id = list(index[word].keys())
    vlb_index[word] += vlb(doc_id[0])
    for i in range(1,len(doc_id)):
        gap = doc_id[i] - doc_id[i-1]
        vlb_index[word] += vlb(gap)



##bigrams

allbigrams = []
for i in range(len(english)):
    list_of_words = word_tokenize(english['all'].iloc[i])
    for word in list_of_words:
        if word not in stop_words:
            allbigrams += [word[i:i+2] for i in range(0, len(word)-1)]
allbigrams = set(allbigrams)

bigramindex = {x: [] for x in allbigrams}

for i in range(len(english)):
    list_of_words = word_tokenize(english['all'].iloc[i])
    for j in range(len(list_of_words)):
        word = list_of_words[j]
        if word not in stop_words:
            bigrams = [word[i:i+2] for i in range(0, len(word)-1)]
            for k in bigrams:
                if word not in bigramindex[k]:
                        bigramindex[k].append(word)
#print(bigramindex)

def jaccard(a, b):
    intersection = list(set(a) & set(b))
    union = list(set().union(a,b))
    return len(intersection) / len(union)


##query_edit
def query_edit(query):
    query = query.lower()
    query = remove_punctuations(query)
    query = re.sub("\d+", "", query)
    candids = []
    query_bigrams = [query[i:i+2] for i in range(0, len(query)-1)]
    for bigram in query_bigrams:
        for candid in bigramindex[bigram]:
            candid_bigrams = [candid[i:i+2] for i in range(0, len(candid)-1)]
            if jaccard(candid_bigrams, query_bigrams) > 0.3 :
                candids.append(candid)
    best = 1000000
    edited_query = []
    candids = list(set(candids))
    for candid in candids:
        if nltk.edit_distance(candid,query) < best:
            best = nltk.edit_distance(candid,query)
    for candid in candids:
        if nltk.edit_distance(candid, query) == best:
            edited_query.append(candid)
    return list(set(edited_query))


print(query_edit("thas"))



##search

##idf
idf_word = [0 for i in range(len(dic))]

for i in range(len(dic)):
    word = dic[i]
    idf_word[i] = math.log(len_doc / len(index[word].keys()))
print(idf_word)

tf = [[0 for i in range (len(dic))] for j in range(len_doc)]

for i in range(len(dic)):
    word = dic[i]
    posting_word = index[word]
    for key in posting_word.keys():
        tf[key][i] = len(posting_word[key])

w_t_d = [[0 for i in range (len(dic))] for j in range(len_doc)]

w_t_d = preprocessing.normalize(w_t_d, norm='l2')


for i in range(len_doc):
    for j in range(len(dic)):
        w_t_d[i][j] = idf_word[j] * tf[i][j]


def query_search_vector(query):

    query = query.lower()
    query = remove_punctuations(query)
    query = re.sub("\d+", "",query)

    list_of_words = word_tokenize(query)
    list_of_words = [i for i in list_of_words if not i in stop_words]


    for i in range(len(list_of_words)):
        if list_of_words[i] not in dic:
            list_of_words[i] = query_edit(list_of_words[i])[0]

    vector = [0 for i in range(len(dic))]
    for j in range(len(list_of_words)):
        for i in range(len(dic)):
            if dic[i] == list_of_words[j]:
                vector[i] += 1
    vector /= np.linalg.norm(vector)
    similarities = [0 for i in range(len_doc)]
    for i in range(len_doc):
        similarities[i] = np.dot(vector, w_t_d[i])

    most = np.array(similarities).argsort()[-10:][::-1]
    removes = []
    for doc_index in most:
        flag = False
        for query_word in list_of_words:
            if tf[doc_index][dic.index(query_word)] != 0:
                flag = True
        if flag == False:
            removes.append(doc_index)
    return [i for i in most if i not in removes]



print(query_search_vector("Parmalat to"))


def find_window(post, k):
	alls = []
	index = 0
	for l in post:
		for i in l:
			alls.append((i, index))
		index += 1
	alls.sort()
	mark = [0 for i in range(len(post))]
	en, cnt = 0, 0
	for st in range(len(alls)):
		while en < len(alls) and alls[en][0] - alls[st][0] <= k:
			mark[alls[en][1]] += 1
			if mark[alls[en][1]] == 1:
				cnt += 1
			en += 1
		if cnt == len(post):
			return True
		mark[alls[st][1]] -= 1
		if mark[alls[st][1]] == 0:
			cnt -= 1
	return False

def query_search_proxmity(query):

    query = query.lower()
    query = remove_punctuations(query)
    query = re.sub("\d+", "",query)

    list_of_words = word_tokenize(query)
    list_of_words = [word for word in list_of_words if not word in stop_words]
    list_of_words = [lemma.lemmatize(word) for word in list_of_words]
    list_of_words = [word if word in dic else query_edit(word) for word in list_of_words]


    intersect_all = index[list_of_words[0]].keys()

    for query_word in list_of_words:
        intersect_all = list(set(intersect_all) & set(index[query_word].keys()))

    A = []
    if len(intersect_all) == 0:
        print("No document found")
    else:
        for doc in intersect_all:
            all_positions = []
            for query_word in list_of_words:
                all_positions.append(index[query_word][doc])
                if find_window(all_positions,2 * len(list_of_words)):
                    A.append(doc)

    vector = [0 for i in range(len(dic))]
    for j in range(len(list_of_words)):
        for i in range(len(dic)):
            if dic[i] == list_of_words[j]:
                vector[i] += 1
    vector /= np.linalg.norm(vector)

    similarities = [0 for i in range(A)]
    for i in range(A):
        similarities[i] = np.dot(vector, w_t_d[A[i]])
    return np.array(similarities).argsort()[-10:][::-1]






