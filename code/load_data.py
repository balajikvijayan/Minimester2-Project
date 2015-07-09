import pandas as pd
import numpy as np
import os
from pymongo import MongoClient

class LoadData:
    def __init__(self):
        pass
    
    def LoadData(self, datapath, dbname, collname):
        #Path of data files
        pospath =  datapath+r'\pos'
        negpath = datapath+r'\neg'
        #PyMongo variables
        client = MongoClient()
        db = client[dbname]
        collection = db[collname]
        #Data lists
        review = []
        opinion = []
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
        df.head()
        result = collection.insert(df.T.to_dict().values())
        return "{} Records loaded".format(collection.count())
    
if __name__ == '__main__':
    loader = LoadData()
    datapath = r"C:\Anaconda\Galvanize\Minimester2-Project\txt_sentoken"
    dbname = 'reviews'
    collname = 'movies'
    print loader.LoadData(datapath, dbname, collname)
