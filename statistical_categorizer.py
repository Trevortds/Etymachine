import tsvopener
import csv
import math
import time
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
import sklearn
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import pickle
import numpy as np
# import matplotlib as plt
import pylab as pl
import pandas as pd

# set these to true and run whenever categorized.tsv changes
initletters = False
initwords = False
initsyllables = True
new_design_matrix = False
verbose = True
iofilename = "bow_"

stops = stopwords.words('english')
stops.append("English")
stops.append("French")
stops.append("Norse")
stops.append("Latin")
stops.append("Greek")

etymdict = tsvopener.etymdict
category_dict = {}

etymwords = []  #list of words in the order that they appear in categorized.tsv
syllables = []  #list of syllables present based on the words 
                #as they appear in the above list

with open('categorized.tsv', 'r') as readfile:
    etymreader = csv.reader(readfile, delimiter='\t', quotechar='|')

    

    for line in etymreader:
        category_dict[line[0]] = line[1]
        etymwords.append(line[0])

with open('categorized.tsv.syllables', 'r') as readfile:

    for line in readfile:
        syllables.append(line.split())



allwords = []
allletters = []
allsyllables = []

print("Initializing wordlist")

if initwords:
    allwords = set()
    # start off as sets to prevent duplicates
    for item in etymdict.keys():
        for word in word_tokenize(etymdict[item]):
            if word in stops:
                continue
            else:
                allwords.add(word)

    allwords = list(allwords)
    with open("words.txt", 'w') as writefile:
        for word in allwords:
            writefile.write(word + "\n")
else:
    with open("words.txt", "r") as readfile:
        for line in readfile:
            allwords.append(line[:-1])

print("Initializing letter list")

if initletters:
    allletters = set()
    for word in allwords:
        for letter in word:
            allletters.add(letter)

    allletters = list(allletters)
    with open("letters.txt", 'w') as writefile:
        for letter in allletters:
            writefile.write(letter + "\n")
else:
    with open("letters.txt", 'r') as readfile:
        for line in readfile:
            allletters.append(line[:-1])


print("Initializing syllable list")

if initsyllables:
    allsyllables = set()

    for item in syllables:
        # each of these is a list of syllables present
        for syl in item:
            allsyllables.add(syl)

    allsyllables = list(allsyllables)

    with open("syllables.txt", 'w') as writefile:
        for syl in allsyllables:
            writefile.write(syl + "\n")
else:
    with open("syllables.txt", 'r') as readfile:
        for line in readfile:
            allsyllables.append(line[:-1])


def syllable_extractor(word):
    output = [0] * len(allsyllables)

    for syl in syllables[etymwords.index(word)]:
        output[allsyllables.index(syl)] = 1

    return output


def year_extractor(definition):
    '''
    returns the first year that appears in the definiton, rounded to the
        century. Returns 0 if none found
    :definition: the definiton of the word being featurized
    :return: 2-digit century
    '''
    four_match = re.search("\d?\d\d\d", definition)
    two_match = re.search("(\d?\d)(th|st|rd|nd)", definition)
    if four_match is None and two_match is None:
        # uncomment for verbatim
        return [0]
        # uncomment for one-hot 
        # return [0] * 20
    elif two_match is None:
        match = int(four_match.group(0))
        match = match // 100
    else:
        match = int(two_match.group(1))

    # uncomment for verbatim 
    return [match]

    # uncomment for one-hot
    # output = [0] * 20
    # output[match] = 1
    # return output


def featurizer(word, definition, bow=True, letters=False, 
    year=False, syllables=False):
    '''
    Transforms  an etymological entry into a feature vector
    :word: the word being featurized
    :definition: the definition of the word in etymonline
    :letters: whether to add the "characters present" feature
    :year: whether to add the "first attested century" feature
    :return: a bag of words sparse vector
    '''
    output = []
    bowout = []
    yearout = []
    lettersout = []
    syllout = []
    if bow:
        bowout = [0] * (len(allwords))
    if letters:
        lettersout = [0] * (len(allletters))
    if year:
        yearout = year_extractor(definition)
    if syllables:
        #insert syllable logic here
        syllout = syllable_extractor(word)

    # initialize output vector
    for definition_word in word_tokenize(definition):
        # check what letters are present
        if letters:
            for char in definition_word:
                lettersout[allletters.index(char)] = 1
        # add to bag of words
        if bow and definition_word not in stops:
            bowout[allwords.index(definition_word)] += 1

    output = bowout+lettersout+yearout+syllout
    return csr_matrix(output)


def get_matrices(test_percent, iofilename, bow, letters, year,
                 syllables, verbose, new_design_matrix):
    '''
    Prepares data for analysis. 
    :test_percent: if new_design_matrix is set to "true", how much of the 
        data available to put into the X and t matrices (set to 1 for full
        analysis, and .12 for making sure that things work)
    :iofilename: The name of the file to be read from 
        (if new_design_matrix=False)
        or the file to print the results into (if new_design_matrix=True)
    :letters: whether to include "characters present" feature
    :year: whether to include "first attested century" feature
    :verbose: whether to print updates on progress or not 
        (recommended if making new design matrices for the whole dataset)
    :return: X, t (design matrix and targets)
    '''
    X = []
    t = []

    if new_design_matrix:
        if verbose:
            print("constructing input vectors")

        num_test_keys = math.floor(len(category_dict) * test_percent)

        keys_to_test = list(category_dict.keys())[:num_test_keys]

        time_start = time.time()

        for i, key in enumerate(keys_to_test):
            if category_dict[key] == "English":
                t.append(0)
            elif category_dict[key] == "French":
                t.append(1)
            elif category_dict[key] == "Norse":
                t.append(2)
            elif category_dict[key] == "Latin":
                t.append(3)
            elif category_dict[key] == "Greek":
                t.append(4)
            elif category_dict[key] == "Other":
                t.append(5)
            else:
                # for some reason the regex categorizer failed to categorize
                # some items, we will skip these
                continue
            X.append(featurizer(key, etymdict[key], bow=bow, letters=letters,
                                year=year, syllables=syllables))
            if verbose:
                if i % (num_test_keys // 100) == 0:
                    print(("done {:d} out of {:d} words. " +
                           "Elapsed time: {:f}").format(i,
                                                        num_test_keys,
                                                        time.time() -
                                                        time_start))
        X = vstack(X, format='csr')
        with open(iofilename+"design.pkl", "wb") as writefile:
            pickle.dump(X, writefile)

        with open(iofilename+"targets.pkl", "wb") as writefile:
            pickle.dump(t, writefile)
    else:
        if verbose:
            print("Reading in design and target matrices")
        with open(iofilename+"design.pkl", "rb") as readfile:
            X = pickle.load(readfile)
        with open(iofilename+"targets.pkl", "rb") as readfile:
            t = pickle.load(readfile)

    return X, t


##########################################
# From Stack Exchange 
# http://stackoverflow.com/questions/23339523/sklearn-cross-validation-with-multiple-scores
# needed to be updated to modern sklearn structure, there is no more
# cross_validation.Kfolds, and the function of KFolds changed. 


def get_true_and_pred_CV(estimator, X, y, n_folds, cv, params):
    ys = []
    for train_idx, valid_idx in cv.split(X):
        clf = estimator
        clf.fit(X[train_idx], y[train_idx])
        cur_pred = clf.predict(X[valid_idx])
        # if isinstance(X, np.ndarray):
        #     clf.fit(X[train_idx], y[train_idx])
        #     cur_pred = clf.predict(X[valid_idx])
        # elif isinstance(X, pd.DataFrame):
        #     clf.fit(X.iloc[train_idx, :], y[train_idx]) 
        #     cur_pred = clf.predict(X.iloc[valid_idx, :])
        # else:
        #     raise Exception('Only numpy array and pandas DataFrame ' \
        #                     'as types of X are supported')

        ys.append((y[valid_idx], cur_pred))
    return ys


def fit_and_score_CV(estimator, X, y, n_folds=10, stratify=True, **params):
    if not stratify:
        cv_arg = sklearn.model_selection.KFold(n_folds)
    else:
        cv_arg = sklearn.model_selection.StratifiedKFold(y, n_folds)

    ys = get_true_and_pred_CV(estimator, X, y, n_folds, cv_arg, params)    
    print(ys)
    cv_acc = list(map(lambda tp: sklearn.metrics.accuracy_score(tp[0], tp[1]), ys))
    print(cv_acc)
    cv_pr_weighted = list(map(lambda tp: sklearn.metrics.precision_score(tp[0], tp[1], average='weighted'), ys))
    print(cv_pr_weighted)
    cv_rec_weighted = list(map(lambda tp: sklearn.metrics.recall_score(tp[0], tp[1], average='weighted'), ys))
    print(cv_rec_weighted)
    cv_f1_weighted = list(map(lambda tp: sklearn.metrics.f1_score(tp[0], tp[1], average='weighted'), ys))
    print(cv_f1_weighted)

    # the approach below makes estimator fit multiple times
    #cv_acc = sklearn.cross_validation.cross_val_score(algo, X, y, cv=cv_arg, scoring='accuracy')
    #cv_pr_weighted = sklearn.cross_validation.cross_val_score(algo, X, y, cv=cv_arg, scoring='precision_weighted')
    #cv_rec_weighted = sklearn.cross_validation.cross_val_score(algo, X, y, cv=cv_arg, scoring='recall_weighted')   
    #cv_f1_weighted = sklearn.cross_validation.cross_val_score(algo, X, y, cv=cv_arg, scoring='f1_weighted')
    return {'CV accuracy': np.mean(cv_acc), 'CV precision_weighted': np.mean(cv_pr_weighted),
            'CV recall_weighted': np.mean(cv_rec_weighted), 'CV F1_weighted': np.mean(cv_f1_weighted)}






############################################3










def run_CV_test(test_percent, iofilename, bow=True, letters=False,
                year=False, syllables=False,
                verbose=True, new_design_matrix=False):
    '''
    Runs a cross-validation test on the data with the given parameters. 
    :test_percent: if new_design_matrix is set to "true", how much of the 
        data available to put into the X and t matrices (set to 1 for full
        analysis, and .12 for making sure that things work)
    :iofilename: The name of the file to be read from 
        (if new_design_matrix=False)
        or the file to print the results into (if new_design_matrix=True)
    :letters: whether to include "characters present" feature
    :year: whether to include "first attested century" feature
    :verbose: whether to print updates on progress or not 
        (recommended if making new design matrices for the whole dataset)
    '''
    X, t = get_matrices(test_percent, iofilename, bow, letters, year,
                        syllables, verbose, new_design_matrix)

    if verbose:
        print("Performing 5-fold CV with a linear SVM")

    clf = svm.SVC(kernel='linear')

    scores = fit_and_score_CV(clf, X, np.asarray(t), n_folds=5, stratify=False)

    print("Accuracy: {:.02f}".format(scores["CV accuracy"]))
    print("Precision: {:.02f}".format(scores["CV precision_weighted"]))
    print("Recall: {:.02f}".format(scores["CV recall_weighted"]))
    print("F-score: {:.02f}".format(scores["CV F1_weighted"]))


    # score = cross_val_score(clf, X, t, cv=5, scoring="precision_macro")

    # print(score)
    # print("Precision: {:0.2f} (+/-) {:0.2f}".format(score.mean(),
    #                                                score.std()*2))
    # score = cross_val_score(clf, X, t, cv=5, scoring="recall_macro")

    # print(score)
    # print("Recall: {:0.2f} (+/-) {:0.2f}".format(score.mean(),
    #                                                score.std()*2))

    # score = cross_val_score(clf, X, t, cv=5, scoring="accuracy")

    # print(score)
    # print("Accuracy: {:0.2f} (+/-) {:0.2f}".format(score.mean(),
    #                                                score.std()*2))
    
    # score = cross_val_score(clf, X, t, cv=5, scoring="f1_macro")

    # print(score)
    # print("F1: {:0.2f} (+/-) {:0.2f}".format(score.mean(),
    #                                                score.std()*2))


new_design_matrix = False
# print("bow only test")
# run_CV_test(1, "full_bow_", letters=False, year=False, verbose=True)

# print("bow and letters")
# run_CV_test(1, "full_bow_letters_", letters=True, year=False, verbose=True)

# print("bow letters and year")
# run_CV_test(1, "full_bow_letters_year_", letters=True, year=True,
#             verbose=True)

verbose = True


def makelinearmodels(filename, holdout_percent, normalize_X=False,
                     test_percent=1, bow=True, letters=False, year=False, 
                     syllables=False,
                     new_design_matrix=False):
    '''
    Trains a classifier on the data from the given filename and generates a 
        confusion matrix on it. 
    :filename: The name of the file to be read from 
        (if new_design_matrix=False)
        or the file to print the results into (if new_design_matrix=True)
    :holdout_percent: What percent of the total data available to hold out 
        for testing
    :normalize_X: Whether to normalize the feature space
    :return: sklearn.svm.LinearSVC classifier trained on the given data
    '''
    X, t = get_matrices(test_percent, filename, bow=bow, letters=letters, 
                        year=year, syllables=syllables,
                        verbose=True, 
                        new_design_matrix=new_design_matrix)

    holdout = math.floor(X.shape[0] * holdout_percent)
    linclf = svm.LinearSVC()

    labels = ["English", "French", "Norse", "Latin", "Greek", "Other"]
    labels_digits = [0, 1, 2, 3, 4, 5]

    if normalize_X:
        print("normalizing")
        X = normalize(X, norm="l2", axis=0)

    trn = X[:-holdout]
    tst = X[-holdout:]
    t_trn = t[:-holdout]
    t_tst = t[-holdout:]

    # linearSVC ################################
    print("training linearSVC")
    linclf.fit(trn, t_trn)

    print("done")
    predicted = linclf.predict(tst)
    print("metrics.f1score: ")
    print(metrics.f1_score(t_tst, predicted, labels_digits, average="macro"))

    cm = metrics.confusion_matrix(t_tst, predicted, labels_digits)

    fig = pl.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    if normalize_X:
        pl.title("LinearSVC classifier on " + filename +
                 "design.pkl (normalized)")
    else:
        pl.title("LinearSVC classifier on " + filename + "design.pkl")
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    pl.xlabel('Predicted')
    pl.ylabel('True')

    # pl.show()

    return linclf


# makelinearmodels("full_bow_", .1)
# makelinearmodels("full_bow_letters_", .1)
# clf = makelinearmodels("full_bow_letters_year_", .1)


# makelinearmodels("12bow_", 200, normalize_X=True)
# makelinearmodels("12bow_letters_", 200, normalize_X=True)
# makelinearmodels("full_bow_letters_year_", .1, normalize_X=True)

# pl.show()


# classify everything, including the data that didn't have a classification yet
# new_category_dict = {}
# transformer = {
#     0: "English",
#     1: "French",
#     2: "Norse",
#     3: "Latin",
#     4: "Greek",
#     5: "Other",
# }

# for word in etymdict.keys():
#     vector = featurizer(word, etymdict[word], letters=True, year=True)
#     prediction = clf.predict(vector)
#     prediction = prediction[0]
#     new_category_dict[word] = transformer[prediction]
