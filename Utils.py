'''
Created on July 20, 2018
@author : hsiaoyetgun (yqxiao)
'''
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf
import numpy as np
import os
from datetime import timedelta
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import pickle
import time
import json
import re

UNKNOWN = '<UNK>'
PADDING = '<PAD>'
CATEGORIE_ID = {'entailment' : 0, 'neutral' : 1, 'contradiction' : 2}

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

# print tensor shape
def print_shape(varname, var):
    """
    :param varname: tensor name
    :param var: tensor variable
    """
    print('{0} : {1}'.format(varname, var.get_shape()))

# init embeddings randomly
def init_embeddings(vocab, embedding_dims):
    """
    :param vocab: word nums of the vocabulary
    :param embedding_dims: dimension of embedding vector
    :return: randomly init embeddings with shape (vocab, embedding_dims)
    """
    rng = np.random.RandomState(None)
    random_init_embeddings = rng.normal(size = (len(vocab), embedding_dims))
    return random_init_embeddings.astype(np.float32)

# load pre-trained embeddings
def load_embeddings(path, vocab):
    """
    :param path: path of the pre-trained embeddings file
    :param vocab: word nums of the vocabulary
    :return: pre-trained embeddings with shape (vocab, embedding_dims)
    """
    with open(path, 'rb') as fin:
        _embeddings, _vocab = pickle.load(fin)
    embedding_dims = _embeddings.shape[1]
    embeddings = init_embeddings(vocab, embedding_dims)
    for word, id in vocab.items():
        if word in _vocab:
            embeddings[id] = _embeddings[_vocab[word]]
    return embeddings.astype(np.float32)

# normalize the word embeddings
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis = 1).reshape((-1, 1))
    return embeddings / norms

# count the number of trainable parameters in model
def count_parameters():
    totalParams = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variableParams = 1
        for dim in shape:
            variableParams *= dim.value
        totalParams += variableParams
    return totalParams

# time cost
def get_time_diff(startTime):
    endTime = time.time()
    diff = endTime - startTime
    return timedelta(seconds = int(round(diff)))

# build vocabulary according the training data
def build_vocab(dataPath1,dataPath2, vocabPath, threshold = 0, lowercase = True):
    """
    :param dataPath: path of training data file
    :param vocabPath: path of vocabulary file
    :param threshold: mininum occurence of vocabulary, if a word occurence less than threshold, discard it
    :param lowercase: boolean, lower words or not
    """
    cnt = Counter()
    with open(dataPath1, mode='r', encoding='utf-8') as iF:
        for line in iF:
            try:
                if lowercase:
                    line = line.lower()
                tempLine = line.strip().split('||')
                l1 = tempLine[1][:-1]
                l2 = tempLine[2][:-1]
                words1 = l1.split(' ')
                for word in list(words1):
                    cnt[word] += 1
                words2 = l2.split(' ')
                for word in list(words2):
                    cnt[word] += 1
            except:
                pass
    with open(dataPath2, mode='r', encoding='utf-8') as iF:
        for line in iF:
            try:
                if lowercase:
                    line = line.lower()
                tempLine = line.strip().split('||')
                l1 = tempLine[1][:-1]
                l2 = tempLine[2][:-1]
                words1 = l1.split(' ')
                for word in list(words1):
                    cnt[word] += 1
                words2 = l2.split(' ')
                for word in list(words2):
                    cnt[word] += 1
            except:
                pass

    cntDict = [item for item in cnt.items() if item[1] >= threshold]
    cntDict = sorted(cntDict, key=lambda d: d[1], reverse=True)
    wordFreq = ['||'.join([word, str(freq)]) for word, freq in cntDict]
    with open(vocabPath, mode='w', encoding='utf-8') as oF:
        oF.write('\n'.join(wordFreq) + '\n')
    print('Vacabulary is stored in : {}'.format(vocabPath))

# load vocabulary
def load_vocab(vocabPath, threshold = 0):
    """
    :param vocabPath: path of vocabulary file
    :param threshold: mininum occurence of vocabulary, if a word occurence less than threshold, discard it
    :return: vocab: vocabulary dict {word : index}
    """
    vocab = {}
    reverse_vocab = {}
    index = 2
    vocab[PADDING] = 0
    vocab[UNKNOWN] = 1
    reverse_vocab[0]=PADDING
    reverse_vocab[1]=UNKNOWN
    with open(vocabPath, encoding='utf-8') as f:
        for line in f:
            items = [v.strip() for v in line.split('||')]
            if len(items) != 2:
                print('Wrong format: ', line)
                continue
            if items[0]!= "":
                word, freq = items[0], int(items[1])
                if freq >= threshold:
                    vocab[word] = index
                    reverse_vocab[index]=word
                    index += 1
    return vocab,reverse_vocab

# data preproceing, convert words into indexes according the vocabulary
def sentence2Index(dataPath, vocabDict, maxLen = 50, lowercase = True):
    """
    :param dataPath: path of data file
    :param vocabDict: vocabulary dict {word : index}
    :param maxLen: max length of sentence, if a sentence longer than maxLen, cut off it
    :param lowercase: boolean, lower words or not
    :return: s1Pad: padded sentence1
             s2Pad: padded sentence2
             s1Mask: actual length of sentence1
             s2Mask: actual length of sentence2
    """
    s1List, s2List, labelList = [], [], []
    s1Mask, s2Mask = [], []
    with open(dataPath, mode='r', encoding='utf-8') as f:
        for line in f:
            try:
                l, s1, s2 = [v.strip() for v in line.strip().split('||')]
                if lowercase:
                    s1, s2 = s1.lower(), s2.lower()
                s1 = [v.strip() for v in s1.split()]
                s2 = [v.strip() for v in s2.split()]
                if len(s1) > maxLen:
                    s1 = s1[:maxLen]
                if len(s2) > maxLen:
                    s2 = s2[:maxLen]
                if l in CATEGORIE_ID:
                    labelList.append([CATEGORIE_ID[l]])
                    s1List.append([vocabDict[word] if word in vocabDict else vocabDict[UNKNOWN] for word in s1])
                    s2List.append([vocabDict[word] if word in vocabDict else vocabDict[UNKNOWN] for word in s2])
                    s1Mask.append(len(s1))
                    s2Mask.append(len(s2))
            except:
                ValueError('Input Data Value Error!')

    s1Pad, s2Pad = pad_sequences(s1List, maxLen, padding='post'), pad_sequences(s2List, maxLen, padding='post')
    s1MaskList, s2MaskList = (s1Pad > 0).astype(np.int32), (s2Pad > 0).astype(np.int32)
    enc = OneHotEncoder(sparse=False)
    labelList = enc.fit_transform(labelList)
    s1Mask = np.asarray(s1MaskList, np.int32)
    s2Mask = np.asarray(s2MaskList, np.int32)
    labelList = np.asarray(labelList, np.int32)
    return s1Pad, s1Mask, s2Pad, s2Mask, labelList

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# def sentence2Index(dataPath, vocabDict, maxLen = 40, lowercase = True):
#     """
#     :param dataPath: path of data filea
#     :param vocabDict: vocabulary dict {word : index}
#     :param maxLen: max length of sentence, if a sentence longer than maxLen, cut off it
#     :param lowercase: boolean, lower words or not
#     :return: s1Pad: padded sentence1
#              s2Pad: padded sentence2
#              s1Mask: actual length of sentence1
#              s2Mask: actual length of sentence2
#     """
#     s1List, s2List, labelList = [], [], []
#     s1Mask, s2Mask = [], []
#     with open(dataPath, mode='r', encoding='utf-8') as f:
#         c1,c2,c3=0,0,0
#         for line in f:
#             try:
#                 l, s1, s2 = [v.strip() for v in line.strip().split('||')]
#                 if lowercase:
#                     s1, s2 = s1.lower(), s2.lower()
#                 s1 = [v.strip() for v in s1.split()]
#                 s2 = [v.strip() for v in s2.split()]
                
#                 if len(s1) > maxLen:
#                     s1 = s1[:maxLen]
#                 if len(s2) > maxLen:
#                     s2 = s2[:maxLen]
                    
                
#                 if len(s2)-len(s1) > 10 :
#                     c1+=1
#                     s2Chunks = chunks(s2,len(s1))
#                     for s in s2Chunks:
#                         #print("Did this happen")
#                         if l in CATEGORIE_ID:
#                             labelList.append([CATEGORIE_ID[l]])
#                             s1List.append([vocabDict[word] if word in vocabDict else vocabDict[UNKNOWN] for word in s1])
#                             s2List.append([vocabDict[word] if word in vocabDict else vocabDict[UNKNOWN] for word in s])
#                             s1Mask.append(len(s1))
#                             s2Mask.append(len(s))
                            
#                 elif len(s1)-len(s2) > 10:
#                     c2+=1
#                     s1Chunks = chunks(s1,len(s2))
#                     for s in s1Chunks:
#                         if l in CATEGORIE_ID:
#                             labelList.append([CATEGORIE_ID[l]])
#                             s1List.append([vocabDict[word] if word in vocabDict else vocabDict[UNKNOWN] for word in s])
#                             s2List.append([vocabDict[word] if word in vocabDict else vocabDict[UNKNOWN] for word in s2])
#                             s1Mask.append(len(s1))
#                             s2Mask.append(len(s2))
#                 else:
#                     c3+=1
#                     if l in CATEGORIE_ID:
#                         labelList.append([CATEGORIE_ID[l]])
#                         s1List.append([vocabDict[word] if word in vocabDict else vocabDict[UNKNOWN] for word in s1])
#                         s2List.append([vocabDict[word] if word in vocabDict else vocabDict[UNKNOWN] for word in s2])
#                         s1Mask.append(len(s1))
#                         s2Mask.append(len(s2))
#             except:
#                 ValueError('Input Data Value Error!')
#     print(c1,c2,c3)
#     s1Pad, s2Pad = pad_sequences(s1List, maxLen, padding='post'), pad_sequences(s2List, maxLen, padding='post')
#     s1MaskList, s2MaskList = (s1Pad > 0).astype(np.int32), (s2Pad > 0).astype(np.int32)
#     enc = OneHotEncoder(sparse=False)
#     labelList = enc.fit_transform(labelList)
#     s1Mask = np.asarray(s1MaskList, np.int32)
#     s2Mask = np.asarray(s2MaskList, np.int32)
#     labelList = np.asarray(labelList, np.int32)
#     return s1Pad, s1Mask, s2Pad, s2Mask, labelList

# generator : generate a batch of data
def next_batch(premise, premise_mask, hypothesis, hypothesis_mask, y, batchSize = 64, shuffle = True):
    """
    :param premise_mask: actual length of premise
    :param hypothesis_mask: actual length of hypothesis
    :param shuffle: boolean, shuffle dataset or not
    :return: generate a batch of data (premise, premise_mask, hypothesis, hypothesis_mask, label)
    """
    sampleNums = len(premise)
    batchNums = int((sampleNums - 1) / batchSize) + 1

    if shuffle:
        indices = np.random.permutation(np.arange(sampleNums))
        premise = premise[indices]
        premise_mask = premise_mask[indices]
        hypothesis = hypothesis[indices]
        hypothesis_mask = hypothesis_mask[indices]
        y = y[indices]

    for i in range(batchNums):
        startIndex = i * batchSize
        endIndex = min((i + 1) * batchSize, sampleNums)
        yield (premise[startIndex : endIndex], premise_mask[startIndex : endIndex],
               hypothesis[startIndex : endIndex], hypothesis_mask[startIndex : endIndex],
               y[startIndex : endIndex])

# convert SNLI dataset from json to txt (format : gold_label || sentence1 || sentence2)
def convert_data(jsonPath, txtPath):
    """
    :param jsonPath: path of SNLI dataset file
    :param txtPath: path of output
    """
    fout = open(txtPath, 'w')
    with open(jsonPath) as fin:
        i = 0
        cnt = {key : 0 for key in CATEGORIE_ID.keys()}
        cnt['-'] = 0
        for line in fin:
            text = json.loads(line)

            evidence = text['evidence']
            evidence = re.sub('[\']','',evidence)
            evidence = re.sub('[^A-z0-9 -]',' ',evidence).lower()
            evidence = ' '.join(evidence.split())
            evidence=evidence.replace(" s "," ")

            claim = text['claim']
            claim = re.sub('[\']','',claim)
            claim = re.sub('[^A-z0-9 -]',' ',claim).lower()
            claim = ' '.join(claim.split())
            claim = claim.replace(" s "," ")

            
            cnt[text['label']] += 1
            print('||'.join([text['label'], evidence, claim]), file = fout)

            i += 1
            if i % 10000 == 0:
                print(i)

    for key, value in cnt.items():
        print('#{0} : {1}'.format(key, value))
    print('Source data has been converted from "{0}" to "{1}".'.format(jsonPath, txtPath))

# convert embeddings from txt to format : (embeddings, vocab_dict)
def convert_embeddings(srcPath, dstPath):
    """
    :param srcPath: path of source embeddings
    :param dstPath: path of output
    """
    vocab = {}
    id = 0
    wrongCnt = 0
    with open(srcPath, 'r', encoding = 'utf-8') as fin:
        lines = fin.readlines()
        wordNums = len(lines)
        line = lines[0].strip().split()
        vectorDims = len(line) - 1
        embeddings = np.zeros((wordNums, vectorDims), dtype = np.float32)
        for line in lines:
            items = line.strip().split()
            if len(items) != vectorDims + 1:
                wrongCnt += 1
                print(line)
                continue
            if items[0] in vocab:
                wrongCnt += 1
                print(line)
                continue
            vocab[items[0]] = id
            embeddings[id] = [float(v) for v in items[1:]]
            id += 1

        embeddings = embeddings[0 : id, ]
        with open(dstPath, 'wb') as fout:
            pickle.dump([embeddings, vocab], fout)

        print('valid embedding nums : {0}, embeddings shape : {1},'
              ' wrong format embedding nums : {2}, total embedding nums : {3}'.format(len(vocab),
                                                                                      embeddings.shape,
                                                                                      wrongCnt,
                                                                                      wordNums))
        print('Original embeddings has been converted from {0} to {1}'.format(srcPath, dstPath))

# print log info on SCREEN and LOG file simultaneously
def print_log(*args, **kwargs):
    print(*args)
    if len(kwargs) > 0:
        print(*args, **kwargs)
    return None

# print all used hyper-parameters on both SCREEN an LOG file
def print_args(args, log_file):
    """
    :Param args: all used hyper-parameters
    :Param log_f: the log life
    """
    argsDict = vars(args)
    argsList = sorted(argsDict.items())
    print_log("------------- HYPER PARAMETERS -------------", file = log_file)
    for a in argsList:
        print_log("%s: %s" % (a[0], str(a[1])), file = log_file)
    print("-----------------------------------------", file = log_file)
    return None

if __name__ == '__main__':
    # dataset preprocessing
    # if not os.path.exists('./SNLI/clean data/'):
    #     os.makedirs('./SNLI/clean data/')

    #convert_data('../../../Datasets/train_data/train_final.json', './data/clean_data/train_fever.txt')
    #convert_data('../../../Datasets/train_data/dev_final.json', './data/clean_data/dev_fever.txt')
    # convert_data('../../../Datasets/dev_data/test_data_final.json', './data/clean_data/test_fever.txt')

    # embedding preprocessing
    # convert_embeddings('../../../Datasets/glove/glove.42B.300d.txt', './data/embeddings_42B.pkl')

    # vocabulary preprocessing
    build_vocab('./data/clean_data/train_fever_update.txt','./data/clean_data/damTest.txt' ,'./data/clean_data/vocab_fever2.txt')
