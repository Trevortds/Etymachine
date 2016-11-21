import tsvopener
import csv
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn import svm

# set these to true and run whenever categorized.tsv changes
initletters = True
initwords = True

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
            allwords.append(line)

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
            allletters.append(line)


def bag_of_words_featurizer(word, definition):
    output = [0] * len(allwords)
    # initialize output vector
    for definition_word in word_tokenize(definition):
        if definition_word not in stops:
            output[allwords.index(definition_word)] += 1

    return output


X = []
t = []

print("constructing input vectors")

keys_to_test = list(category_dict.keys())[:100]


for key in keys_to_test:
    X.append(bag_of_words_featurizer(key, etymdict[key]))
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
        t.append(5)
