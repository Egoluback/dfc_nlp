import os, gensim, logging, wget, zipfile
import pandas as pd
import numpy as np

from pymystem3 import Mystem
from collections import Counter

# from functools import lru_cache

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

mapping = {'A': 'ADJ', 'ADV': 'ADV', 'ADVPRO': 'ADV', 'ANUM': 'ADJ', 'APRO': 'DET', 'COM': 'ADJ', 'CONJ': 'SCONJ', 'INTJ': 'INTJ', 'NONLEX': 'X', 'NUM': 'NUM', 'PART': 'PART', 'PR': 'ADP', 'S': 'NOUN', 'SPRO': 'PRON', 'UNKN': 'X', 'V': 'VERB'}

prepositions = ['в', 'шт', 'кг', 'без', 'до', 'из', 'к', 'на', 'по', 'о', 'от', 'перед', 'при', 'через', 'с', 'у', 'за', 'над', 'об', 'под', 'про', 'для', 'вблизи', 'вглубь', 'вдоль', 'возле', 'около', 'вокруг', 'впереди', 'после', 'посредством', 'путём', 'насчёт', 'поводу', 'ввиду', 'случаю', 'течение', 'благодаря', 'несмотря на', 'спустя']

class PreProcess:
    def __init__(self, train, noreply = False):
        if (noreply): self.train_noreply = train[train.category_id != -1].drop_duplicates('item_name')
        else: self.train_noreply = train
        
        self.mystem = Mystem()

        self.frequent = []
    
    def model_load(self, download = False):
        udpipe_url = 'https://rusvectores.org/static/models/udpipe_syntagrus.model'
        model_url = 'http://vectors.nlpl.eu/repository/11/180.zip'
        
        if (download): m_ = wget.download(model_url)
        
        model_file = model_url.split('/')[-1]
        with zipfile.ZipFile("data/" + model_file, 'r') as archive:
            stream = archive.open('model.bin')
            self.model_vv = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True)
    
    def frequent_save(self):
        self.frequent.to_csv("data/frequent.csv")
    
    def frequent_load(self):
        self.frequent = pd.read_csv("data/frequent.csv")
    
    def fit(self, el_count = None, cat_load = False):
        self.model_load()
        print("--- model load completed ---")
        
        if (not cat_load):
            self.cat_words()
            self.frequent_save()
        else: self.frequent_load()
        print("--- categories frequent assembled ---")
        
        if (el_count is not None): return self.analyze(self.train_noreply.item_name.iloc[: el_count], self.model_vv, vec = True)
        else: return self.analyze(self.train_noreply.item_name, self.model_vv, vec = True)
    
    def tag_mystem(self, text):
        processed = self.mystem.analyze(text)
        tagged = []
        for w in processed:
            try:
                lemma = w["analysis"][0]["lex"].lower().strip()
                pos = w["analysis"][0]["gr"].split(',')[0]
                pos = pos.split('=')[0].strip()
                
                if pos in mapping: tagged.append(lemma + '_' + mapping[pos]) # здесь мы конвертируем тэги
                else: tagged.append(lemma + '_X') # на случай, если попадется тэг, которого нет в маппинге
            except KeyError: continue # пропускаем знаки препинания
            except IndexError: continue
        return tagged
    
    def cat_words(self):
        book = {}

        for id in self.train_noreply['category_id'].unique():
            df = pd.DataFrame(self.train_noreply[self.train_noreply['category_id'] == id]['item_name'])
            count = Counter(" ".join(df["item_name"]).lower().split()).most_common(20)
            result = ""
            
            if not count: continue
            
            for j in count:
                n = j[0]
                n_mystem = self.tag_mystem(n)
                
                if (not n_mystem): continue
                else: n_mystem = n_mystem[0]

                if (len(n) < 3 or n in prepositions or not n.isalpha()) or n_mystem not in self.model_vv: continue

                result = n_mystem
                break

            book[id] = result

        self.frequent = pd.DataFrame.from_dict(book, orient='index')
        
        self.frequent.loc[118] = self.tag_mystem("Новогодний")[0]
        self.frequent.loc[79] = self.tag_mystem("Мясо")[0]
        self.frequent.loc[67] = self.tag_mystem("Суши")[0]
        self.frequent.loc[13] = self.tag_mystem("Автомобиль")[0]
        self.frequent.loc[42] = self.tag_mystem("Косметика")
        self.frequent.loc[9] = self.tag_mystem("Топливо")[0] # тут, в основном, названия видов топлива, так что word2vec сработает плохо

        # NEW
        self.frequent.loc[71] = self.tag_mystem("Еда из ресторана")[0]
        self.frequent.loc[73] = self.tag_mystem("Рыба, замороженное мясо, тесто, пельмени")[0]
    
    def analyze(self, train, model, vec = False):
        connected_words = self.frequent[0].values

        vectors = list(map(model.word_vec, connected_words))

        train = self.delete90(train)
        train = train.apply(lambda x: self.fin(x, model, vectors, connected_words, train[train == x].index[0] / train.shape[0], vec))

        return pd.DataFrame(list(train.values))

    def delete_between_symbols(self, s1, s2, list_symbol):
        words = (s1, s2)
        if s1 in list_symbol and s2 in list_symbol:
            return list_symbol[: list_symbol.find(words[0]) + len(words[0]) - 1] + list_symbol[1 + list_symbol.rfind(words[1]): ]
        else:
            return list_symbol

    def delete90(self, train):
        return train.apply(lambda x: self.delete_between_symbols('(', ')', x))
    
    def fin(self, string, model, vectors, connected_words, progress, vec = False):
#         print("------")
#         print(string)
        # if ((progress * 100) % 2 == 0): print("Current progress: {0}".format(progress))
        print("Current progress: {0}".format(progress))
        words = self.tag_mystem(string)
        word = self.choose_relevant(words, vectors, connected_words, model)
        if not vec:
            if word in model:
                return word
            else:
                return 'нет слова'
        else:
            if word in model:
                return model.word_vec(word)
            else:
                return np.array([0] * model.vector_size)
    
    def choose_relevant(self, words, vectors, connected_words, model):
        best_word = None
        max_score = -1
        best_pair = None
        ms = 0
        for i, word in enumerate(words):
            for v in connected_words:
                if word in model:
                    c_d = model.similarity(word, v) / (i + 5)
                    if c_d > max_score:
                        best_word = word
                        max_score = c_d
                        ms = c_d * (i + 5)
                        best_pair = v
#         print("Best pair is " + str(best_pair) + " with " + str(best_word) + ". Score is " + str(ms))

        return best_word


if (__name__ == '__main__'):
    train = pd.read_csv("data/cleared_data.csv")

    train.drop(train[train.item_name.isna()].index[0], axis = 0, inplace = True)

    model_preprocess = PreProcess(train, noreply = True)

    vectors = model_preprocess.fit(el_count = None, cat_load = False)
    
    vectors.to_csv('data/vectors_n.csv')