import pandas as pd
import numpy as np
import os
from pymongo import MongoClient

class LoadData:
    def __init__(self):
        pass
    
    def LoadData(self, datapath, dbname, collname):
        #Path of data files
        pospath, negpath =  datapath+r'\pos', datapath+r'\neg'
        #PyMongo variables
        client = MongoClient()
        db = client[dbname]
        collection = db[collname]
        #Data lists
        review, opinion = [], []
        #Drop the existing MongoDB data so duplicates aren't loaded
        collection.drop()
        #loop through neg files
        for name in os.listdir(negpath):
            #context manager so files aren't floating around
            with open(negpath+'\\'+name, 'r') as fo:
                #read in the data
                data = fo.read()
                #remove apostrophes so that contractions are treated as one word
                #otherwise wouldn't, can't, won't etc will all tokenize to t
                review.append(data.replace("'",""))
                #Label!
                opinion.append(False)
        
        #loop through pos files
        for name in os.listdir(pospath):
            #context manager so files aren't floating around
            with open(pospath+'\\'+name, 'r') as fo:
                #read in the data
                data = fo.read()
                #remove apostrophes so that contractions are treated as one word
                #otherwise wouldn't, can't, won't etc will all tokenize to t
                review.append(data.replace("'",""))
                #Label!
                opinion.append(True)
        
        df = pd.DataFrame(zip(review,opinion))
        df.columns = ['Review','Opinion']
        result = collection.insert(df.T.to_dict().values())
        return "{} Review records loaded".format(collection.count())

    def LoadBingLiuSentiment(self, sentimentpath, dbname, collname):
        #Path of data files
        pospath, negpath = sentimentpath+r'\positive-words.txt', sentimentpath+r'\negative-words.txt'
        #PyMongo variables
        client = MongoClient()
        db = client[dbname]
        collection = db[collname]
        #Data lists
        words, sentiment, pos, neg = [],[],[],[]
        #Drop the existing MongoDB data so duplicates aren't loaded
        collection.drop()
        #loop through files
        for name in [pospath, negpath]:
            #context manager so files aren't floating around
            with open(name, 'r') as fo:
                #read in the data
                data = fo.read()
                u_data = data.decode('cp1252') 
                utf8_data = u_data.encode("utf8").split('\n')        
                #Label!
                if name == pospath:
                    pos.extend(utf8_data)
                if name == negpath:
                    neg.extend(utf8_data)
        
        for word in list(set(pos) & set(neg)):
            #de-dupe words in both pos and neg lexicon
            del pos[pos.index(word)]
            del neg[neg.index(word)]
        
        #now we build the proper sentiment and word lists to load
        sentiment.extend([1]*len(pos))
        sentiment.extend([-1]*len(neg))
        words.extend(pos)
        words.extend(neg)
        #Dataframes for insertion of sentiment data into MongoDB
        df = pd.DataFrame(zip(words,sentiment))
        df.columns = ['Word','Sentiment']
        result = collection.insert(df.T.to_dict().values())
        return "{} Sentiment records loaded".format(collection.count())

    def LoadMPQASentiment(self, path, dbname, collname):
        #PyMongo variables
        client = MongoClient()
        db = client[dbname]
        collection = db[collname]
        #Drop the existing MongoDB data so duplicates aren't loaded
        collection.drop()
        #read the mpqa sentiment into a DF
        sentiment_df = pd.read_csv(path, sep=' ')
        #column cleanup
        sentiment_df.columns = ['Type','Len', 'Word1', 'Pos1','S',\
                                'PriorPolarity']
        #text cleanup IS THERE A BETTER WAY TO DO THIS?
        sentiment_df.iloc[:, 0] = sentiment_df.iloc[:, 0].str.replace('type=', '')
        sentiment_df.iloc[:, 1] = sentiment_df.iloc[:, 1].str.replace('len=', '')
        sentiment_df.iloc[:, 2] = sentiment_df.iloc[:, 2].str.replace('word1=', '')
        sentiment_df.iloc[:, 3] = sentiment_df.iloc[:, 3].str.replace('pos1=', '')
        sentiment_df.iloc[:, 4] = sentiment_df.iloc[:, 4].str.replace('stemmed1=', '')
        sentiment_df.iloc[:, 5] = sentiment_df.iloc[:, 5].str.replace('priorpolarity=', '')
        
        result = collection.insert(sentiment_df.T.to_dict().values())
        return "{} Sentimenet records loaded".format(collection.count())
    
if __name__ == '__main__':
    loader = LoadData()
    print loader.LoadData(r"C:\Anaconda\Galvanize\Minimester2-Project\data\txt_sentoken", 'reviews', 'movies')
    print loader.LoadBingLiuSentiment(r"C:\Anaconda\Galvanize\Minimester2-Project\data\sentiment", 'sentiment', 'bingliu')
    print loader.LoadMPQASentiment(r"C:\Anaconda\Galvanize\Minimester2-Project\data\sentiment\subjclueslen1-HLTEMNLP05.tff", 'sentiment', 'mpqa')