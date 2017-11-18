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

