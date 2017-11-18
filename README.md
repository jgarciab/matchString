# Database merging and string matching
Javier Garcia-Bernardo, 2017

[CODE AND FIGURES HERE: match_strings.ipynb](match_strings.ipynb)

TODO:
- Make it more elegant/flexible, this is recycled code from many years ago.
- Big data to avoid comparing all names in database 1 to all names in database 2. This can be achieved neatly with LSH forests (see
[see here for the current implementation](lsh_forest.py)


## Requirements:
1. Libraries
```
pip install distance numpy pandas matplotlib sklearn seaborn python-Levenshtein 
```

2. Train and test set: Two files with three columns (string1, string2, 0/1 for match)



## How to run it:
- Check [the notebook](match_strings.ipynb)
```
database1 = "./D/database_1.csv"
database2 = "./D/database_2.csv"
train_data_file = "./D/train.csv"
test_data_file = "./D/test.csv"

tfidf_matrix_train,dictTrain,tfidf_matrix_trainBigrams,dictTrainBigrams,lenGram = createTFIDF(database1,database2)
clf,clf2 = train(train_data_file,tfidf_matrix_train,dictTrain,tfidf_matrix_trainBigrams,dictTrainBigrams,lenGram,sep="\t")
predict = test(test_data_file,tfidf_matrix_train,dictTrain,tfidf_matrix_trainBigrams,dictTrainBigrams,lenGram,clf,clf2,sep="\t")
plot(predict)
```

- You can then use clf (the SVM) to predict matches between any two strings, you can use the plot with ROC curve to set up your threshold (or let the algorithm find it, but that will depend on your training set).
```
distances = find_distances(st1,st2)
clf.decision_function(np.array(temp,dtype=float))
```

