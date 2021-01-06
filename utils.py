import re
import os
import torch
import pickle
import random
import numpy as np
from soynlp.hangle import decompose
from keras.preprocessing import text, sequence


# 결과 재생산을 위한 시드값 고정을 위한 함수
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# 한글 자모 분리 코드
# reference - https://github.com/ratsgo/embedding/blob/master/preprocess/unsupervised_nlputils.py
def jamo_sentence(sent):

    doublespace_pattern = re.compile('\s+')

    def transform(char):
        if char == ' ':
            return char
        cjj = decompose(char)
        try:
            len(cjj)
        except:
            return ' '
        if len(cjj) == 1:
            return cjj
        cjj_ = ''.join(c if c != ' ' else '' for c in cjj)
        return cjj_

    sent_ = ''.join(transform(char) for char in sent)
    sent_ = doublespace_pattern.sub(' ', sent_)
    return sent_


# 토큰화된 단어(자모 분리된)가 사전학습된 fasttext 딕셔너리에 있을 경우 해당 임베딩 벡터를 불러오고
# 만약에 사전에 없을 경우 unkown_words에 없을 경우 생략한다
def build_matrix(word_index, word2vec_vocab, max_features, vector_size=200):
    embedding_matrix = np.zeros((max_features + 1, vector_size))
    unknown_words = []
    
    for word, i in word_index.items():
        if i <= max_features:
            try:
                embedding_matrix[i] = word2vec_vocab[word]
            except:
                unknown_words.append(word)
                
    return embedding_matrix, unknown_words


def preprocess_text(input_text):

    with open('./pytorch_model/train_tokenized.pkl', 'rb') as f:
        train_tokenized = pickle.load(f)

    test_jamo_splited = jamo_sentence(input_text)
    max_features = 283820 # 학습시 설정
    tokenizer = text.Tokenizer(num_words = max_features, filters='')
    tokenizer.fit_on_texts(list(train_tokenized) + list(test_jamo_splited))

    # 사전 학습한 Fasttext 임베딩 load
    with open('./pytorch_model/fasttext_vocab.pkl', 'rb') as f:
        embedding_vocab = pickle.load(f)

    # 구축한 vocab를 활용하여 embedding vector 형성
    embedding_matrix, unknown_words = build_matrix(tokenizer.word_index, embedding_vocab, max_features, vector_size=200)

    maxlen = len(test_jamo_splited)
    zeros = np.zeros(maxlen)
    test_tokenized = np.concatenate(tokenizer.texts_to_sequences(test_jamo_splited))
    zeros[-len(test_tokenized):] = test_tokenized
    test_padded = torch.from_numpy(zeros).reshape(1, -1)

    return test_padded, embedding_matrix