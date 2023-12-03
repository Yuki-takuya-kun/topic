import string
import json
import yaml

import sqlite3
import numpy as np
import gensim
from tqdm import tqdm
#from extract.preprocess import sentence_normalize


class LDA_Analyzer:
    def __init__(self, airline):
        with open('config/analyzer.yaml', 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.airline = airline
        #self.NE = spacy.load("en_core_web_sm")
        #self.translator = str.maketrans('', '', string.punctuation)
        
        self.query_sql = f"SELECT Review FROM {self.airline}"
        
        if self.cfg['website'] == 'all' or self.cfg['website'] == 'tripadvisor':
            self.tripconn = sqlite3.connect("dataset/tripadvisor_topic.db")
            self.tripcur = self.tripconn.cursor()
        if self.cfg['website'] == 'all' or self.cfg['website'] == 'skytrax':
            self.skyconn = sqlite3.connect("dataset/skytrax_topic.db")
            self.skycur = self.skyconn.cursor()
            
        # self.lda_model = gensim.models.ldamodel.LdaModel()
        
        
    def analyze(self):
        reviews = []
        topics_dic = {}
        # extract and normalize the review in tripadvisor
        if self.cfg['website'] == 'all' or self.cfg['website'] == 'tripadvisor':
            self.tripcur.execute(self.query_sql)
            for elem in tqdm(self.tripcur.fetchall(), desc="tripadvisor"):
                reviews.append(elem[0].split(" "))
        # extract and normalize the review in tripadvisor
        if self.cfg['website'] == 'all' or self.cfg['website'] == 'skytrax':
            self.skycur.execute(self.query_sql)
            for elem in tqdm(self.skycur.fetchall(), desc="skytrax"):
                reviews.append(elem[0].split(" "))
        dictionary = gensim.corpora.Dictionary(reviews)
        corpus = [dictionary.doc2bow(document) for document in reviews]
        
        
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)
        topics = ldamodel.show_topics(num_topics=-1, num_words=10, formatted=False)
        for topic in topics:
            distribution = {}
            topic_idx = topic[0]
            topic_dis = topic[1]
            for dis in topic_dis:
                distribution[dis[0]] = float(dis[1])
            topics_dic[topic_idx] = distribution
        with open(f"output/lda_{self.cfg['website']}_{self.airline}.json", 'w', encoding='utf-8') as f:
            json.dump(topics_dic, f, indent=4)
            
    def compute_coherence(self):
        reviews = []
        coherences = []
        # extract and normalize the review in tripadvisor
        if self.cfg['website'] == 'all' or self.cfg['website'] == 'tripadvisor':
            self.tripcur.execute(self.query_sql)
            for elem in tqdm(self.tripcur.fetchall(), desc="tripadvisor"):
                reviews.append(elem[0].split(" "))
        # extract and normalize the review in tripadvisor
        if self.cfg['website'] == 'all' or self.cfg['website'] == 'skytrax':
            self.skycur.execute(self.query_sql)
            for elem in tqdm(self.skycur.fetchall(), desc="skytrax"):
                reviews.append(elem[0].split(" "))
        dictionary = gensim.corpora.Dictionary(reviews)
        corpus = [dictionary.doc2bow(document) for document in reviews]
        
        for topic_num in tqdm(range(5, 20)):
            tmp_lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=topic_num, id2word=dictionary, passes=15)
            coherence_model_lda = gensim.models.coherencemodel.CoherenceModel(model=tmp_lda_model, texts=reviews, dictionary=dictionary, coherence='c_npmi')
            coherence_socre = coherence_model_lda.get_coherence()
            coherences.append(coherence_socre)
            del tmp_lda_model, coherence_model_lda
        
        print(coherences)
        np.save(f"output/lda_coherence_{self.cfg['website']}_{self.airline}", np.array(coherences))
        
            
            
