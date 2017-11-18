import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import LSHForest
import time
import pickle
import os

class Matcher():
    def __init__(self,folderSaveData,token_lenght=3):
        #create directory to save data
        self.folderSaveData = folderSaveData

        if not os.path.exists(folderSaveData):
            os.makedirs(folderSaveData)
            
    def tokenize(self,listNames,lenToken=3):
        #string to tokens of size 3
        return [' '.join([name[i:i+lenToken] for i in range(len(name)-2)]) for name in listNames]

    def create_TDIDF(self,trainGrams):
        #Create TDIDF from n-tokens
        self.TF = TfidfVectorizer()
        self.tfidfs = self.TF.fit_transform(trainGrams)


    def fit(self,listNames,variableName):
        #LSHForest. only once for the main database
        self.lshf = LSHForest(random_state=42,n_estimators=50,n_candidates=500)
        self.create_TDIDF(self.tokenize(listNames))
        self.lshf.fit(self.tfidfs)        
        self.listNames = listNames
        pickle.dump(self.lshf,open("{0}/{1}_lshf.dump".format(self.folderSaveData,variableName),"wb+"))
        pickle.dump(listNames,open("{0}/{1}_listNames.dump".format(self.folderSaveData,variableName),"wb+"))
        pickle.dump(self.TF,open("{0}/{1}_TF.dump".format(self.folderSaveData,variableName),"wb+"))
        

    def predict(self,variableName,list_names_to_match):
        with open(self.folderSaveData+variableName+"_matched.csv","w+") as fOUT:
            print("Number of names to match", len(list_names_to_match))
            
            tokenMatch = self.tokenize(list_names_to_match)

            try: 
                tdidf_transformed = self.TF.transform(tokenMatch)
            except:
                self.lshf = pickle.load(open("{0}/{1}_lshf.dump".format(self.folderSaveData,variableName),"rb"))
                self.TF = pickle.load(open("{0}/{1}_TF.dump".format(self.folderSaveData,variableName),"rb"))
                self.listNames = pickle.load(open("{0}/{1}_listNames.dump".format(self.folderSaveData,variableName),"rb"))
                tdidf_transformed = self.TF.transform(tokenMatch)


            print("Finding neighbors")
            t = time.time()
            distances_, indices_ = self.lshf.kneighbors(tdidf_transformed,n_neighbors=100)
            print("Neighbors saved in {:2.2f}".format(time.time()-t))
            
            print("Saving results")
            t = time.time()
            for i,name_to_match in enumerate(list_names_to_match):
                if i%1000 == 0: print("Number matched ", i)
                distances = distances_[i,:]
                indices = indices_[i,:]
        
                name,distances,indices = self.filter_data_exact(name_to_match,distances,indices)
                names_matched = [self.listNames[index] for index in indices]
                string_to_save = "{0}\t{1}\n".format(name,"\t".join([str(_[0])+"\t"+str(_[1]) for _ in zip(names_matched,distances)]))
                fOUT.write(string_to_save)
            print("Results saved in {:2.2f}".format(time.time()-t))
                
    def filter_data_exact(self,name_match,distances,indices,trainingData = None):
        #still to code, for now taking the top 10 matches        
        if trainingData: 
            return name_match, distances[:100],indices[:100] #fancy stuff goes here
        else:
            return name_match, distances[:100],indices[:100]



    def normalizeList(self,listNames_,patterns=[r"\."]):
        """
        Watch out when using it, only works in the format "firstname lastname, shit_we_dont_care_about". The format "lastname, firstname" will break Internet
        """
        import re
        listNames = [_.lower().strip() for _ in listNames_]

        for pattern in patterns:
            pat = re.compile(pattern)
            listNames = [re.sub(pat,"",_) for _ in listNames]

        return listNames
    
path_to_save_results = "./D/"
list_names_orbis = open("./D/database_1.csv").readlines()
list_names_match = open("./D/database_2.csv").readlines()
patterns = [r"\."] #patterns for the normalization (this deletes periods)


M = Matcher(path_to_save_results,token_lenght=3)


#Only once to create the forest
if 1: 
    list_names_train = M.normalizeList(list_names_orbis,patterns=patterns) #Orbis
    M.fit(list_names_train,"movies")

#Match
list_names_match = M.normalizeList(list_names_match,patterns=patterns)
M.predict("movies",list_names_match)

