import yaml
import sqlite3
import spacy
import json

from tqdm import tqdm
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from bertopic import BERTopic
import numpy as np

#from extract.preprocess import sentence_normalize


class Bertopic_analyzer:
    def __init__(self, airline):
        with open("config/analyzer.yaml", 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        self.model = BERTopic()
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
        
            
    def analyze(self):
        reviews = []
        topics_dic = {}
        # extract and normalize the review in tripadvisor
        if self.cfg['website'] == 'all' or self.cfg['website'] == 'tripadvisor':
            self.tripcur.execute(self.query_sql)
            for elem in tqdm(self.tripcur.fetchall(), desc="tripadvisor"):
                reviews.append(elem[0])
        # extract and normalize the review in tripadvisor
        if self.cfg['website'] == 'all' or self.cfg['website'] == 'skytrax':
            self.skycur.execute(self.query_sql)
            for elem in tqdm(self.skycur.fetchall(), desc="skytrax"):
                reviews.append(elem[0])
        topics, probs = self.model.fit_transform(reviews)
        topic_infos = self.model.get_topic_info()
        for key, val in topic_infos['Topic'].items():
            distirbution = {}
            if val >= 0:
                for word, dis in self.model.get_topic(val):
                    distirbution[word] = dis
                topics_dic[val] = distirbution
            
        with open(f"output/bertopic_{self.cfg['website']}_{self.airline}.json", 'w', encoding='utf-8') as f:
            json.dump(topics_dic, f, indent=4)
            
    def compute_coherence(self):
        coherence_scores = []
        reviews = []
        reviews_tokened = []
        
        # extract and normalize the review in tripadvisor
        if self.cfg['website'] == 'all' or self.cfg['website'] == 'tripadvisor':
            self.tripcur.execute(self.query_sql)
            for elem in tqdm(self.tripcur.fetchall(), desc="tripadvisor"):
                tokens = elem[0].split(" ")
                reviews.append(" ".join(tokens))
                reviews_tokened.append(tokens)
        # extract and normalize the review in tripadvisor
        if self.cfg['website'] == 'all' or self.cfg['website'] == 'skytrax':
            self.skycur.execute(self.query_sql)
            for elem in tqdm(self.skycur.fetchall(), desc="skytrax"):
                tokens = elem[0].split(" ")
                reviews.append(" ".join(tokens))
                reviews_tokened.append(tokens)
        dictionary = Dictionary(reviews_tokened)
        corpus = [dictionary.doc2bow(words) for words in reviews_tokened]
        
        for topic_num in tqdm(range(5, 20)):
            model = BERTopic(nr_topics=topic_num)
            
            topics, probs = model.fit_transform(reviews)
            topic_infos = model.get_topic_info()
            # print(topic_infos)
            topics = []
            for i, val in topic_infos['Representation'].items():
                if i > 0:
                    topics.append(val)
            coherence_model = CoherenceModel(topics=topics, texts=reviews_tokened, corpus=corpus, dictionary=dictionary, coherence='c_npmi')
            score = coherence_model.get_coherence()
            coherence_scores.append(score)
        print(coherence_scores)
        np.save(f"output/bertopic_coherence_{self.cfg['website']}_{self.airline}", np.array(coherence_scores))
        
            
    # def compute
            
            
        
        