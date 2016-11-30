# import requests
import re
# import os
# import subprocess
import nltk
import pylab as pl
import tsvopener
import dawg
from nltk.corpus import brown

test_text = "four score and seven years ago, our fathers brought fourth\
             on this continent a new nation."
tokenized_text = nltk.word_tokenize(test_text)

new_category_dict = tsvopener.open_tsv("stat_categorized.tsv")

category_dawg = dawg.BytesDAWG([(x, str.encode(new_category_dict[x])) for x
                                in new_category_dict.keys()])


def make_lexicon_pie(input_dict, title):
    e = 0
    f = 0
    n = 0
    l = 0
    g = 0
    o = 0

    for word in input_dict.keys():
        label = input_dict[word]
        if label == "English":
            e += 1
        elif label == "French":
            f += 1
        elif label == "Norse":
            n += 1
        elif label == "Latin":
            l += 1
        elif label == "Greek":
            g += 1
        else:
            o += 1

    total = e + f + n + l + g + o
    fracs = [o/total, n/total, g/total, l/total, f/total, e/total]
    labels = 'Other', 'Norse', 'Greek', 'Latin', 'French', 'English'
    pl.figure(figsize=(6, 6))
    pl.axes([0.1, 0.1, 0.8, 0.8])
    pl.pie(fracs, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    pl.title(title)
    pl.show()


def make_analysis_pie(sentences, title, token=False, ignore_unknowns=False,
                      show=True):
    e = f = n = g = l = o = u = 0
    if token:
        already_seen = []
    unknowns = []
    for sentence in sentences:
        for word, tag in sentence:
            if token:
                if word in already_seen:
                    continue
                else:
                    already_seen.append(word)
            label = label_word(word, tag)
            if label == "Unknown":
                label = label_word(word.lower(), tag)
            if label == "English":
                e += 1
            elif label == "French":
                f += 1
            elif label == "Norse":
                n += 1
            elif label == "Latin":
                l += 1
            elif label == "Greek":
                g += 1
            elif label == "Other":
                o += 1
            elif label == "Unknown":
                unknowns.append((word, tag))
                u += 1
    total = u + e + f + n + l + g + o
    fracs = [u/total, o/total, n/total, g/total, l/total, f/total, e/total]
    labels = 'Unknown', 'Other', 'Norse', 'Greek', 'Latin', 'French', 'English'
    colors = 'r', 'orange', 'b', 'c', 'm', 'y', 'g'
    if ignore_unknowns:
        total = e + f + n + l + g + o
        fracs = [o/total, n/total, g/total, l/total, f/total, e/total]
        labels = 'Other', 'Norse', 'Greek', 'Latin', 'French', 'English'
        colors = 'orange', 'b', 'c', 'm', 'y', 'g'
    pl.figure(figsize=(6, 6))
    pl.axes([0.1, 0.1, 0.8, 0.8])
    pl.pie(fracs, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90)
    pl.title(title)
    if show:
        pl.show()

    return unknowns
    # return [e, f, n, l, g, o, u, total]


# make_pie(new_category_dict, "Proportions of etymologies in the lexicon")

def reduce_brown_pos(word, brown_tag):
    skip_tags = ['(', ')', '*', ',', '--', '.', ':', '\'\'', '``', '\'',
                 '.-HL', ',-HL', '(-HL', ')-HL']

    if brown_tag in skip_tags:
        return "skip"
    elif brown_tag.startswith("B"):
        return "v"
    elif brown_tag.startswith("H"):
        return "v"
    elif brown_tag.startswith("V"):
        return "v"
    elif brown_tag.startswith("P"):
        return "pron"
    elif brown_tag.startswith("N"):
        return "n"
    elif brown_tag.startswith("J"):
        return "adj"
    elif brown_tag.startswith("W"):
        return "pron"
    elif brown_tag.startswith("R"):
        return "adv"
    elif brown_tag.startswith("U"):
        return "interj"
    elif brown_tag.startswith("Q"):
        return "adj"
    elif brown_tag.startswith("DO"):
        return "v"
    elif brown_tag.startswith("DT"):
        return "adj"
    elif brown_tag.startswith("I"):
        return "prep"
    elif brown_tag.startswith("CC"):
        return "conj"
    elif brown_tag.startswith("CS"):
        return "conj"
    elif brown_tag.startswith("CD"):
        return "skip"
    elif brown_tag.startswith("M"):
        return "v"
    elif brown_tag.startswith("AP"):
        return "adj"
    elif brown_tag.startswith("FW"):
        return None
    elif brown_tag.startswith("OD"):
        return "adj"
    elif brown_tag.startswith("EX"):
        return "adv.,conj"

    else:
        print(word, " ", brown_tag)
        return None


'''
first, check if the word by itself is in the dictionary
if it's not, then check if there's only one entry with "word ("
if it's not, then take the part of speech and look
if still not, then lemmatize and look
'''

lemmatizer = nltk.stem.WordNetLemmatizer()


def label_word(word, brown_tag, lemmatized=False):
    brown_tag = re.sub("-HL", "", brown_tag)
    brown_tag = re.sub("-TL", "", brown_tag)
    if word in category_dawg:
        return category_dawg[word][0].decode()
    if word.lower() in category_dawg:
        return category_dawg[word.lower()][0].decode()
    if len(category_dawg.keys(word + " (")) > 0:
        return category_dawg[category_dawg.keys(word)[0]][0].decode()

    etymtag = reduce_brown_pos(word, brown_tag)
    if etymtag is None:
        return "Unknown"
    if etymtag == "skip":
        return "skip"

    word_n_tag = word + " (" + etymtag + ".)"
    word_n_tag_n_number = word + " (" + etymtag + ".1)"
    if word_n_tag in category_dawg:
        return category_dawg[word_n_tag][0].decode()
    if word_n_tag_n_number in category_dawg:
        return category_dawg[word_n_tag_n_number][0].decode()

    if lemmatized:
        return "Unknown"

    if etymtag == "n":
        wordnet_tag = "n"
    elif etymtag == "v":
        wordnet_tag = "v"
    elif etymtag == "adj":
        wordnet_tag = "a"
    elif etymtag == "adv":
        wordnet_tag = "v"
    else:
        return "Unknown"

    lemma = lemmatizer.lemmatize(word, pos=wordnet_tag)

    return label_word(lemma, brown_tag, lemmatized=True)


def big_pie_maker():
    sentences = brown.tagged_sents("ca09")
    title = "Words in 1961 Philadelphia Inquirer political article"
    print(title)
    make_analysis_pie(sentences, title, show=False)
    title = "Tokens in 1961 Philadelphia Inquirer political article"
    make_analysis_pie(sentences, title, show=False, token=True)

    sentences = brown.tagged_sents("cm01")
    title = "Words in Robert A Henlein's Stranger in a Strange Land"
    print(title)
    make_analysis_pie(sentences, title, show=False)
    title = "Tokens in Robert A Henlein's Stranger in a Strange Land"
    make_analysis_pie(sentences, title, show=False, token=True)

    sentences = brown.tagged_sents("cp26")
    title = "Words in Evrin D Krause's The Snake"
    print(title)
    make_analysis_pie(sentences, title, show=False)
    title = "Tokens in Evrin D Krause's The Snake"
    make_analysis_pie(sentences, title, show=False, token=True)

    sentences = brown.tagged_sents("cd07")
    title = "Words in Peter Eversveld's Faith Amid Fear"
    print(title)
    make_analysis_pie(sentences, title, show=False)
    title = "Tokens in Peter Eversveld's Faith Amid Fear"
    make_analysis_pie(sentences, title, show=False, token=True)

    sentences = brown.tagged_sents("ch09")
    title = "Words in the Public Laws of the 87th Congress"
    print(title)
    make_analysis_pie(sentences, title, show=False)
    title = "Tokens in the Public Laws of the 87th Congress"
    make_analysis_pie(sentences, title, show=False, token=True)

    sentences = brown.tagged_sents("cj16")
    title = "Words in Nagaraj and Black: Wound-Tumor Virus Antigen"
    print(title)
    make_analysis_pie(sentences, title, show=False)
    title = "Tokens in Nagaraj and Black: Wound-Tumor Virus Antigen"
    make_analysis_pie(sentences, title, show=False, token=True)

    pl.show()


# big_pie_maker()

otherwords = []
for words in new_category_dict.keys():
    if new_category_dict[words] == "Other":
        otherwords.append(words)

print("number of 'other': ", len(otherwords))
import random
random.shuffle(otherwords)
print(otherwords[:100])
