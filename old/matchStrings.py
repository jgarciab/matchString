__author__ = 'j'
import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import Levenshtein as lev
from sklearn import svm, linear_model
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
import pylab as plt

# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

def customaxis(ax, c_left='k', c_bottom='k', c_right='none', c_top='none', lw=2, size=12, pad=8):
    '''
    From stackoverflow. User gcalmettes
    '''
    for c_spine, spine in zip([c_left, c_bottom, c_right, c_top],
                              ['left', 'bottom', 'right', 'top']):
        if c_spine != 'none':
            ax.spines[spine].set_color(c_spine)
            ax.spines[spine].set_linewidth(lw)
        else:
            ax.spines[spine].set_color('none')
    if (c_bottom == 'none') & (c_top == 'none'): # no bottom and no top
        ax.xaxis.set_ticks_position('none')
    elif (c_bottom != 'none') & (c_top != 'none'): # bottom and top
        ax.tick_params(axis='x', direction='out', width=lw, length=7,
                      color=c_bottom, labelsize=size, pad=pad)
    elif (c_bottom != 'none') & (c_top == 'none'): # bottom but not top
        ax.xaxis.set_ticks_position('bottom')
        ax.tick_params(axis='x', direction='out', width=lw, length=7,
                       color=c_bottom, labelsize=size, pad=pad)
    elif (c_bottom == 'none') & (c_top != 'none'): # no bottom but top
        ax.xaxis.set_ticks_position('top')
        ax.tick_params(axis='x', direction='out', width=lw, length=7,
                       color=c_top, labelsize=size, pad=pad)
    if (c_left == 'none') & (c_right == 'none'): # no left and no right
        ax.yaxis.set_ticks_position('none')
    elif (c_left != 'none') & (c_right != 'none'): # left and right
        ax.tick_params(axis='y', direction='out', width=lw, length=7,
                       color=c_left, labelsize=size, pad=pad)
    elif (c_left != 'none') & (c_right == 'none'): # left but not right
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='y', direction='out', width=lw, length=7,
                       color=c_left, labelsize=size, pad=pad)
    elif (c_left == 'none') & (c_right != 'none'): # no left but right
        ax.yaxis.set_ticks_position('right')
        ax.tick_params(axis='y', direction='out', width=lw, length=7,
                       color=c_right, labelsize=size, pad=pad)

def jaccard(set_1, set_2):
    """
    :param set_1: set of characters string 1
    :param set_2: set of characters string 2
    :return: jaccard distance
    """
    n = len(set_1.intersection(set_2))
    return n / float(len(set_1) + len(set_2) - n)

def dice_coefficient(a,b,lenGram=2):
    """
    :param a: string 1
    :param b: string 2
    :param lenGram: length of the n-grams
    :return: dice score

    From Rossetta code
    """
    if not len(a) or not len(b): return 0.0
    """ quick case for true duplicates """
    if a == b: return 1.0
    """ if a != b, and a or b are single chars, then they can't possibly match """
    if len(a) == 1 or len(b) == 1: return 0.0

    """ use python list comprehension, preferred over list.append() """
    a_bigram_list = [a[i:i+lenGram] for i in range(len(a)-1)]
    b_bigram_list = [b[i:i+lenGram] for i in range(len(b)-1)]

    a_bigram_list.sort()
    b_bigram_list.sort()

    # assignments to save function calls
    lena = len(a_bigram_list)
    lenb = len(b_bigram_list)
    # initialize match counters
    matches = i = j = 0
    while (i < lena and j < lenb):
        if a_bigram_list[i] == b_bigram_list[j]:
            matches += lenGram
            i += 1
            j += 1
        elif a_bigram_list[i] < b_bigram_list[j]:
            i += 1
        else:
            j += 1

    score = float(matches)/float(lena + lenb)
    return score

def createTDIDF():
    ## Bag of words
    with open("./data/movies.csv") as f:
        train_set1 = [line.lower().rstrip() for line in f]
    with open("./data/dvd.csv") as f:
        train_set2 = [line.lower().rstrip() for line in f]

    train_set = sorted(list(set(train_set1 + train_set2)))
    # Create dictionary to find movie
    dictTrain = dict()
    for i,movie in enumerate(train_set):
        dictTrain[movie] = i

    # Find weitghts
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set)

    ## Tri-grams
    lenGram  = 3
    train_setBigrams = []
    for mov in train_set:
        temp = [mov[i:i+lenGram] for i in range(len(mov)-1)]
        temp = [elem for elem in temp if len(elem) == lenGram]
        train_setBigrams.append(' '.join(temp))

    train_setBigrams = sorted(list(set(train_setBigrams)))
    dictTrainBigrams = dict()
    for i,movie in enumerate(train_setBigrams):
        dictTrainBigrams[movie] = i
    tfidf_vectorizerBigrams = TfidfVectorizer()
    tfidf_matrix_trainBigrams = tfidf_vectorizerBigrams.fit_transform(train_setBigrams)

    return [tfidf_matrix_train,dictTrain,tfidf_matrix_trainBigrams,dictTrainBigrams,lenGram]

def cosineBigrams(a,b,dictTrainBigrams,tfidf_matrix_trainBigrams,lenGram=3):
    """
    :param a: string 1
    :param b: string 2
    :param dictTrainBigrams: Dictionary of bigrams  to find index quickly
    :param tfidf_matrix_trainBigrams:  Weigths of bigrrams
    :param lenGram:  Length of n-grams (3)
    :return: cosine similarity (angle between vectors)
    """
    a = a.lower().rstrip()
    b = b.lower().rstrip()
    st1 = ' '.join([elem for elem in [a[i:i+lenGram] for i in range(len(a)-1)] if len(elem) == lenGram])
    st2 = ' '.join([elem for elem in [b[i:i+lenGram] for i in range(len(b)-1)] if len(elem) == lenGram])
    ind_a = dictTrainBigrams[st1]
    ind_b = dictTrainBigrams[st2]
    score = cosine_similarity(tfidf_matrix_trainBigrams[ind_a:ind_a+1], tfidf_matrix_trainBigrams[ind_b:ind_b+1])
    return score

def cosineWords(a,b,dictTrain,tfidf_matrix_train):
    """
    :param a: string 1
    :param b: string 2
    :param dictTrain: Dictionary of wors to find index quickly
    :param tfidf_matrix_train: Weights of words
    :return: cosine similarity (angle between vectors)
    """
    ind_a = dictTrain[a.lower().rstrip()]
    ind_b = dictTrain[b.lower().rstrip()]
    score = cosine_similarity(tfidf_matrix_train[ind_a:ind_a+1], tfidf_matrix_train[ind_b:ind_b+1])
    return score

def train(tfidf_matrix_train,dictTrain,tfidf_matrix_trainBigrams,dictTrainBigrams,lenGram,delete = []):
    allTrainX = list()
    allTrainY = list()
    with open("./data/train.csv") as f:
        for line in f:
            lin = line.split(",")
            if len(lin) == 3:
                st1 = lin[0].lower()
                st2 = lin[1].lower()

                temp = [
                        1.-(lev.distance(st1,st2)*2/(len(st1)+len(st2))),
                        lev.jaro(st1,st2),
                        lev.jaro_winkler(st1,st2),
                        lev.ratio(st1,st2),
                        distance.sorensen(st1,st2),
                        jaccard(set(st1),set(st2)),
                        1. - distance.nlevenshtein(st1,st2,method=1),
                        1. - distance.nlevenshtein(st1,st2,method=2),
                        dice_coefficient(st1,st2,lenGram=2),
                        dice_coefficient(st1,st2,lenGram=3),
                        dice_coefficient(st1,st2,lenGram=4),
                        cosineWords(st1,st2,dictTrain,tfidf_matrix_train),
                        cosineBigrams(st1,st2,dictTrainBigrams,tfidf_matrix_trainBigrams,lenGram)
                    ]
                if len(delete) > 0:
                    for elem in delete:
                        temp[elem] = 0.
                allTrainX.append(temp)
                allTrainY.append(int(lin[2]))


    X = np.array(allTrainX,dtype=float)
    y = np.array(allTrainY,dtype=float)
    clf = svm.LinearSVC(C=1.,dual=False,loss='l2', penalty='l1')
    clf2 = linear_model.LogisticRegression(C=1.,dual=False, penalty='l1')
    clf.fit(X, y)
    clf2.fit(X, y)
    weights = np.array(clf.coef_[0])
    print(weights)
    weights = np.array(clf2.coef_[0])
    print(weights)


    return clf,clf2

def stats(tfidf_matrix_train,dictTrain,tfidf_matrix_trainBigrams,dictTrainBigrams,lenGram,delete = [],plotX=False):
    with open("./data/stats.csv") as infile:
        for i,line in enumerate(infile):
            pass

    dimMatrix = 16
    predict = np.zeros((i+1,dimMatrix))


    clf1,clf2 = train(tfidf_matrix_train,dictTrain,tfidf_matrix_trainBigrams,dictTrainBigrams,lenGram,delete=delete)

    with open("./data/stats.csv") as infile:
        for i,line in enumerate(infile):
            a = line.rstrip().split("\t")

            ## create same vector with more distances
            st1 = a[0].lower()
            st2 = a[1].lower()

            temp = [
            1.-(lev.distance(st1,st2)*2/(len(st1)+len(st2))),
            lev.jaro(st1,st2),
            lev.jaro_winkler(st1,st2),
            lev.ratio(st1,st2),
            distance.sorensen(st1,st2),
            jaccard(set(st1),set(st2)),
            1. - distance.nlevenshtein(st1,st2,method=1),
            1. - distance.nlevenshtein(st1,st2,method=2),
            dice_coefficient(st1,st2,lenGram=2),
            dice_coefficient(st1,st2,lenGram=3),
            dice_coefficient(st1,st2,lenGram=4),
            cosineWords(st1,st2),
            cosineBigrams(st1,st2)]

            if len(delete) > 0:
                for elem in delete:
                    temp[elem] = 0.

            predict[i,:-3] = temp
            predict[i,-3] = clf1.decision_function(np.array(temp,dtype=float))
            predict[i,-2] = clf2.decision_function(np.array(temp,dtype=float))
            predict[i,-1] = a[-1]


    if plotX:
        labelsM = ["Lev","Jaro","Jaro-Winkler","Ratio","Sorensen","Jaccard","Lev1","Lev2","Dice_2","Dice_3","Dice_4","cosineWords","cosineBigrams","SVM","Logit"]
        f1matrix = np.zeros((100,dimMatrix-1))

        fig = plt.figure()
        fig.set_size_inches(9,6)
        ax = fig.add_subplot(111)
        iC = -1
        for i in np.linspace(0,1,100):
            iC += 1
            for j in range(dimMatrix-1):
                t = np.array(predict[:,j])
                if j >= dimMatrix-3:
                    t = (t - np.min(t))/(np.max(t)-np.min(t))
                f1matrix[iC,j] = f1_score(y_pred=t>i ,y_true=predict[:,-1])
        F1scores = []
        for j in range(dimMatrix-1):
            F1scores.append(np.max(f1matrix[:,j]))
            #ax.plot(np.linspace(0,1,100),f1matrix[:,j],label=labelsM[j],color=tableau20[j])
        ax.bar(range(dimMatrix-1),F1scores)
        plt.xticks(np.arange(dimMatrix-1)+0.5,["Lev","Jaro","Jaro-Winkler","Ratio","Sorensen","Jaccard","Lev1","Lev2","Dice_2","Dice_3","Dice_4","cosineWords","cosineBigrams","SVM","Logit"],rotation=45)
        ax.set_ylabel("F1 score")
        ax.set_xlabel("Parameter")
        plt.legend(loc=2)
        customaxis(ax)
        plt.savefig("f1_bar.pdf")
        plt.show()

        fig = plt.figure()
        fig.set_size_inches(9, 6)
        ax = fig.add_subplot(111)

        AUCScores = []
        for j in range(dimMatrix-1):
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(predict[:,-1], predict[:,j])
            AUCScores.append(auc(fpr, tpr))


            # Plot ROC curve
            ax.plot(fpr, tpr, label=labelsM[j],color=tableau20[j])
            ax.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')

        plt.legend(loc=2)
        customaxis(ax)
        plt.savefig("roc.pdf")
        plt.show()

        fig = plt.figure()
        fig.set_size_inches(9, 6)
        ax = fig.add_subplot(111)
        ax.bar(range(dimMatrix-1),AUCScores)
        ax.set_ylabel('Area Under Curve')
        plt.xticks(np.arange(dimMatrix-1)+0.5,["Lev","Jaro","Jaro-Winkler","Ratio","Sorensen","Jaccard","Lev1","Lev2","Dice_2","Dice_3","Dice_4","cosineWords","cosineBigrams","SVM","Logit"],rotation=45)
        customaxis(ax)
        plt.savefig("roc_bar.pdf")
        plt.show()

def main():
    tfidf_matrix_train,dictTrain,tfidf_matrix_trainBigrams,dictTrainBigrams,lenGram = createTDIDF()
    stats(tfidf_matrix_train,dictTrain,tfidf_matrix_trainBigrams,dictTrainBigrams,lenGram,plotX=True)


if __name__ == "__main__":
    main()

