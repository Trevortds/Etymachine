import tsvopener
import csv
import math
import time
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import pickle
# import matplotlib as plt
import pylab as pl

# set these to true and run whenever categorized.tsv changes
initletters = False
initwords = False
new_design_matrix = False
verbose = True
iofilename = "bow_"

stops = stopwords.words('english')
etymdict = tsvopener.etymdict
category_dict = {}

with open('categorized.tsv', 'r') as readfile:
    etymreader = csv.reader(readfile, delimiter='\t', quotechar='|')

    for line in etymreader:
        category_dict[line[0]] = line[1]


allwords = []
allletters = []

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


def year_extractor(definition):
    '''
    returns the first year that appears in the definiton, rounded to the
        century. Returns 0 if none found
    :definition: the definiton of the word being featurized
    :return: 2-digit century
    '''
    match = re.search("\d\d\d\d", definition)
    if match is None:
        return [0]
    match = int(match.group(0))
    match = match // 100
    return [match]


def featurizer(word, definition, letters=False, year=False):
    '''
    Transforms  an etymological entry into a feature vector
    :word: the word being featurized
    :definition: the definition of the word in etymonline
    :letters: whether to add the "characters present" feature
    :year: whether to add the "first attested century" feature
    :return: a bag of words sparse vector
    '''
    output = [0] * (len(allwords))
    yearout = []
    lettersout = []
    if letters:
        lettersout = [0] * (len(allletters))
    if year:
        yearout = year_extractor(definition)
    # initialize output vector
    for definition_word in word_tokenize(definition):
        # check what letters are present
        if letters:
            for char in definition_word:
                lettersout[allletters.index(char)] = 1
        # add to bag of words
        if definition_word not in stops:
            output[allwords.index(definition_word)] += 1
    output = output+lettersout+yearout
    return csr_matrix(output)


def get_matrices(test_percent, iofilename, letters, year, verbose):
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
            X.append(featurizer(key, etymdict[key], letters=letters,
                                year=year))
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


def run_CV_test(test_percent, iofilename, letters=False, year=False,
                verbose=True):
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
    X, t = get_matrices(test_percent, iofilename, letters, year, verbose)

    if verbose:
        print("Performing 10-fold CV with a linear SVM")

    clf = svm.SVC(kernel='linear')
    scores = cross_val_score(clf, X, t, cv=5)

    print(scores)
    print("Accuracy: {:0.2f} (+/-) {:0.2f}".format(scores.mean(),
                                                   scores.std()*2))


new_design_matrix = False
# print("bow only test")
# run_CV_test(1, "full_bow_", letters=False, year=False, verbose=True)

# print("bow and letters")
# run_CV_test(1, "full_bow_letters_", letters=True, year=False, verbose=True)

# print("bow letters and year")
# run_CV_test(1, "full_bow_letters_year_", letters=True, year=True,
#             verbose=True)

verbose = True


def makelinearmodels(filename, holdout_percent, normalize_X=False):
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
    X, t = get_matrices(1, filename, letters=True, year=True,
                        verbose=True)

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
    print(metrics.f1_score(t_tst, predicted, labels_digits, average="micro"))

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
clf = makelinearmodels("full_bow_letters_year_", .1)


# makelinearmodels("12bow_", 200, normalize_X=True)
# makelinearmodels("12bow_letters_", 200, normalize_X=True)
# makelinearmodels("full_bow_letters_year_", .1, normalize_X=True)

# pl.show()


# classify everything, including the data that didn't have a classification yet
new_category_dict = {}
transformer = {
    0: "English",
    1: "French",
    2: "Norse",
    3: "Latin",
    4: "Greek",
    5: "Other",
}

for word in etymdict.keys():
    vector = featurizer(word, etymdict[word], letters=True, year=True)
    prediction = clf.predict(vector)
    prediction = prediction[0]
    new_category_dict[word] = transformer[prediction]
