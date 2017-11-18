__author__ = 'j'

import Levenshtein as lev


def filter():
    import codecs
    import pandas as pd

    DVD = pd.read_csv("dvd_csv.txt",encoding="iso-8859-1")
    dvdSet = set(DVD['DVD_Title'])

    moviesSet = set()
    with codecs.open("movies.txt",encoding="ISO-8859-1") as f:
        for line in f:
            a = line.split("\"")
            if len(a) > 1:
                moviesSet.add(a[1])

    dvdSet = sorted(list(dvdSet))
    moviesSet = sorted(list(moviesSet))

    with open("dvd.csv","w") as outfile:
        outfile.write("\n".join(dvdSet))
    with open("movies.csv","w") as outfile:
        outfile.write("\n".join(moviesSet))

def createTest():
    import Levenshtein as lev

    def distances(st1,st2):
        #if
        return lev.jaro(st1,st2)


    dvds = []
    with open("dvd.csv") as f:
        for i,j in enumerate(f):
            dvds.append(j)

    movies = []
    with open("movies.csv") as f:
        for i,j in enumerate(f):
            movies.append(j)

    dvds = [dvd for dvd in dvds if dvd > "A"]
    movies = [movie for movie in movies if movie > "A"]
    print(len(dvds),len(movies))

    with open("../data/train.csv","w") as f:
        i = 0
        for dvd in dvds:
            prefix = dvd[0]
            i += 1
            maxSimil = 0.
            for movie in movies:
                if movie[0] == prefix:
                    tempSim = distances(dvd,movie)
                    if  tempSim > maxSimil:
                        maxSimil = tempSim
                        maxMovie = movie
            if maxSimil > 0.8:
                print(i,dvd.rstrip(),maxMovie.rstrip())
                f.write("%s,%s,\n" %(dvd.rstrip(),maxMovie.rstrip()))
def test(clf):
    dvds = []
    with open("dvd.csv") as f:
        for i,j in enumerate(f):
            dvds.append(j)

    movies = []
    with open("movies.csv") as f:
        for i,j in enumerate(f):
            movies.append(j)

    dvds = [dvd for dvd in dvds if dvd > "B"]
    movies = [movie for movie in movies if movie > "B"]
    print(len(dvds),len(movies))

    with open("test.csv","w") as f:
        i = 0
        for dvd in dvds:
            prefix = dvd[0]
            i += 1
            maxSimil = 0.
            for movie in movies:
                if movie[0] == prefix:
                    tempSim = lev.jaro(dvd,movie)
                    if  tempSim > maxSimil:
                        maxSimil = tempSim
                        maxMovie = movie

            temp = [
                    1.-(lev.distance(dvd,maxMovie)/len(dvd)),
                    lev.jaro(dvd,maxMovie),
                    lev.jaro_winkler(dvd,maxMovie),
                    lev.ratio(dvd,maxMovie),
                ]
            print("%s\t%s\t%f\t%f" %(dvd.rstrip(),maxMovie.rstrip(),clf.decision_function(temp),clf.predict(temp)))
            f.write("%s\t%s\t%f\t%f\t%f\t%f\t%f\t%i\n" %(dvd.rstrip(),maxMovie.rstrip(),1.-(lev.distance(dvd,maxMovie)/len(dvd)),lev.jaro(dvd,maxMovie),lev.jaro_winkler(dvd,maxMovie),lev.ratio(dvd,maxMovie),clf.decision_function(temp),clf.predict(temp)))


def corroborateBlind():
    with open("../data/stats.csv","a") as outfile:
        with open("test.csv") as f:
            for line in f:
                a = line.split("\t")
                print(a[-1])
                if a[0] > "G" and int(a[-1]) == 1:
                    print("%s\n%s"%(a[0],a[1]))
                    name = input()
                    if name == "" or name == "0":
                        n = "0"
                    elif name == "1":
                        n = "1"
                    else:
                        n = "-9"

                    if n != "-9":
                        outfile.write(line.rstrip()+"\t"+n+"\n")
                    print("\n"*10)

"""


f = open("weightsImportance.csv","a")
f.write("\t".join(["Lev0","Jaro","Jaro_Winkler","Ratio","Sorensen","Jaccard","Lev1","Lev2","Dice2","Dice3","Dice4","CosWords","Cos2",
                   "wLev0","wJaro","wJaro_Winkler","wRatio","wSorensen","wJaccard","wLev1","wLev2","wDice2","wDice3","wDice4","wCosWords","wCos2","intercept",
                   "AUC"])+"\n")
t = np.ones(13)
aucScore,weights,interc = stats(delete=[])
f.write("\t".join([str(int(_)) for _ in t])+"\t"+"\t".join([str(_) for _ in weights])+"\t"+str(interc)+"\t"+str(aucScore)+"\n")

for i in range(13):
    t = np.ones(13)
    t[i] = 0
    aucScore,weights,interc = stats(delete=[i])
    f.write("\t".join([str(int(_)) for _ in t])+"\t"+"\t".join([str(_) for _ in weights])+"\t"+str(interc)+"\t"+str(aucScore)+"\n")
    for j in range(i+1,13):
        print(i,j)
        t = np.ones(13)
        t[i] = 0
        t[j] = 0
        aucScore,weights,interc = stats(delete=[i,j])
        f.write("\t".join([str(int(_)) for _ in t])+"\t"+"\t".join([str(_) for _ in weights])+"\t"+str(interc)+"\t"+str(aucScore)+"\n")
        for h in range(j+1,13):
            t = np.ones(13)
            t[i] = 0
            t[j] = 0
            t[h] = 0
            aucScore,weights,interc = stats(delete=[i,j,h])
            f.write("\t".join([str(int(_)) for _ in t])+"\t"+"\t".join([str(_) for _ in weights])+"\t"+str(interc)+"\t"+str(aucScore)+"\n")


f.close()

[-2.59228719  2.30942469 -0.95178134  0.          0.         -0.08255314
  0.          0.          3.96012396  0.308956    0.          0.87710215
  0.23226918]
[-2.22845174  0.          0.          0.          0.          0.          0.
  0.          3.83980274  2.11160209  0.          2.97987619  1.21379607]
"""
