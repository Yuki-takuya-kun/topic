import yaml
import sqlite3
import json

from tqdm import tqdm
import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from hdbscan import HDBSCAN
from bertopic import BERTopic
import numpy as np
from pprint import pprint

#from extract.preprocess import sentence_normalize


class Analyzer:
    def __init__(self, airline):
        with open("config/analyzer.yaml", 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
            
        self.airline = airline
        
        self.query_sql = f"SELECT Review FROM {self.airline}"
        
        self.reviews = []
        self.get_dataset()
        
    def get_dataset(self):
        if self.cfg['website'] == 'all' or self.cfg['website'] == 'tripadvisor':
            self.tripconn = sqlite3.connect("dataset/tripadvisor_topic.db")
            self.tripcur = self.tripconn.cursor()
        if self.cfg['website'] == 'all' or self.cfg['website'] == 'skytrax':
            self.skyconn = sqlite3.connect("dataset/skytrax_topic.db")
            self.skycur = self.skyconn.cursor()

        # extract and normalize the review in tripadvisor
        if self.cfg['website'] == 'all' or self.cfg['website'] == 'tripadvisor':
            self.tripcur.execute(self.query_sql)
            for elem in tqdm(self.tripcur.fetchall(), desc="tripadvisor"):
                self.reviews.append(elem[0].split(" "))
        # extract and normalize the review in tripadvisor
        if self.cfg['website'] == 'all' or self.cfg['website'] == 'skytrax':
            self.skycur.execute(self.query_sql)
            for elem in tqdm(self.skycur.fetchall(), desc="skytrax"):
                self.reviews.append(elem[0].split(" "))
        
        
        self.dictionary = gensim.corpora.Dictionary(self.reviews)
        self.corpus = [self.dictionary.doc2bow(document) for document in self.reviews]
     
            
    def analyze(self):
        res = []
        
        #bertopic, (model, coherence, diversity)
        bert_models = []
        for co in range(2, 35, 1):
            hdbscan_model = HDBSCAN(min_cluster_size=co, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
            bert_model = BERTopic(hdbscan_model=hdbscan_model)
            #print([" ".join(review) for review in self.reviews])
            topics, probs = bert_model.fit_transform([" ".join(review) for review in self.reviews])
            topic_infos = bert_model.get_topic_info()
            #print(topic_infos)
            topics_dic = {}
            topics = []
            for key, val in topic_infos['Topic'].items():
                distirbution = {}
                topic = []
                if val >= 0:
                    for word, dis in bert_model.get_topic(val):
                        distirbution[word] = dis
                        topic.append(word)
                    topics.append(topic)
                    topics_dic[val] = distirbution
            if len(topics) == 0:
                coherence_score = 0
            else:
                #pprint(topics)
                coherence_model = CoherenceModel(topics=topics, texts=self.reviews, corpus=self.corpus, dictionary=self.dictionary)
                coherence_score = coherence_model.get_coherence()
            
            # compute diversity
            unique_word = set()
            word_num = 0
            for topic_words in topics:
                word_num += len(topic_words)
                unique_word = unique_word.union(set(topic_words))
            if word_num == 0:
                diversity_score = 0
            else:
                diversity_score = len(unique_word) / word_num
            print(f'using bert topic with cluster_size of {co}, coherence: {coherence_score}, diversity: {diversity_score}, topic num: {len(topics)}')
            bert_models.append([topics_dic, coherence_score, diversity_score])
        
        # coherence_avg = sum([coherence[1] for coherence in bert_models]) / len(bert_models)
        # diversity_avg = sum([diversity[2] for diversity in bert_models]) / len(bert_models)
        coherence_scores = [coherence[1] for coherence in bert_models]
        diversity_scores = [diversity[2] for diversity in bert_models]
        coherence_avg = sum(coherence_scores) / len(coherence_scores)
        diversity_avg = sum(diversity_scores) / len(diversity_scores)
        coherence_norms = [(coherence_score - min(coherence_scores)) / (max(coherence_scores) - min(coherence_scores)) for coherence_score in coherence_scores]
        diversity_norms = [(diversity_score - min(diversity_scores)) / (max(diversity_scores) - min(diversity_scores)) for diversity_score in diversity_scores]
        # print(coherence_norms)
        # print(diversity_norms)
        # bert_models.sort(key=lambda x: (x[1] - min(coherence_scores)) / (max(coherence_scores) - min(coherence_scores)) + \
        #                 (x[2] - min(diversity_scores) / (max(diversity_scores) - min(diversity_scores))))
        bert_models.sort(key= lambda x: (x[1] - coherence_avg) + (x[2] - diversity_avg) * 0.1)
        best_bert = bert_models[-1]
        print(f'best bertopic model with coherence:{best_bert[1]}, diversity:{best_bert[2]}')
        print('='*50)
        
        #save to file
        with open(f"output/bertopic_{self.cfg['website']}_{self.airline}.json", 'w', encoding='utf-8') as f:
            json.dump(best_bert[0], f, indent=4)
        res.append([self.airline, 'bertopic', best_bert[1], best_bert[2]])
        
        # lda, (model, coherence, diversity)
        lda_models = []
        for topic_num in range(5, 20, 1):
            ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=topic_num, id2word=self.dictionary, passes=15)
            topics = ldamodel.show_topics(num_topics=-1, num_words=10, formatted=False)
            topics_dic = {}
            topics_words = []
            for topic in topics:
                distribution = {}
                topic_word = []
                topic_idx = topic[0]
                topic_dis = topic[1]
                for dis in topic_dis:
                    distribution[dis[0]] = float(dis[1])
                    topic_word.append(dis[0])
                topics_dic[topic_idx] = distribution
                topics_words.append(topic_word)
                
            coherence_model = CoherenceModel(topics=topics_words, texts=self.reviews, corpus=self.corpus, dictionary=self.dictionary)
            coherence_score = coherence_model.get_coherence()
            
            # compute diversity
            unique_word = set()
            word_num = 0
            for topic_words in topics_words:
                word_num += len(topic_words)
                unique_word = unique_word.union(set(topic_words))
            diversity_score = len(unique_word) / word_num
            print(f'using lda with topic num of {topic_num}, coherence: {coherence_score}, diversity: {diversity_score}')
            lda_models.append([topics_dic, coherence_score, diversity_score])
                
        coherence_scores = [coherence[1] for coherence in lda_models]
        diversity_scores = [diversity[2] for diversity in lda_models]
        coherence_avg = sum(coherence_scores) / len(coherence_scores)
        diversity_avg = sum(diversity_scores) / len(diversity_scores)
        #coherence_norms = [(coherence_score - min(coherence_scores)) / (max(coherence_scores) - min(coherence_scores)) for coherence_score in coherence_scores]
        #diversity_norms = [(diversity_score - min(diversity_scores)) / (max(diversity_scores) - min(diversity_scores)) for diversity_score in diversity_scores]
        
        # lda_models.sort(key=lambda x: (x[1] - min(coherence_scores)) / (max(coherence_scores) - min(coherence_scores)) + \
        #                 (x[2] - min(diversity_scores) / (max(diversity_scores) - min(diversity_scores))))
        lda_models.sort(key= lambda x: (x[1] - coherence_avg) + (x[2] - diversity_avg) * 0.1)
        best_lda = lda_models[-1]
        print(f'best lda model with coherence:{best_lda[1]}, diversity:{best_lda[2]}')
        print('='*100)
        #save to file
        with open(f"output/lda_{self.cfg['website']}_{self.airline}.json", 'w', encoding='utf-8') as f:
            json.dump(best_lda[0], f, indent=4)
        res.append([self.airline, 'lda', best_lda[1], best_lda[2]])
        
        
        return res
        
        
        
            
            
        
        