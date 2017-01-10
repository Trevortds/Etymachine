# import requests
import re
# import os
# import subprocess
import nltk
import pylab as pl
import tsvopener
import dawg
from nltk.corpus import brown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

test_text = "four score and seven years ago, our fathers brought fourth\
             on this continent a new nation."
tokenized_text = nltk.word_tokenize(test_text)

new_category_dict = tsvopener.open_tsv("stat_categorized.tsv")

category_dawg = dawg.BytesDAWG([(x, str.encode(new_category_dict[x])) for x
                                in new_category_dict.keys()])


def make_lexicon_pie(input_dict, title):
    '''
    Make a pie of the full lexicon based on the given mapping of words to 
    languages
    :input_dict: dict of words to language sources
    :title: title to put on the pie chart
    '''
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


def make_analysis_pie(sentences, title="Pie Chart", token=False, ignore_unknowns=False,
                      show=True):
    '''
    Analyzes the given text and generates a pie chart to show the proportions 
    of etymological origins in it. 
    :sentences: tagged sentences from the Brown corpus
    :title: title to go on the chart
    :token: whether to count token frequencies instead of word frequencies
    :ignore_unknowns: whether to have a slice for unknowns in the chart
    :show: whether to show the chart after completeion. 
    :return: the proportions of each language origin in the text in the order
        'Unknown', 'Other', 'Norse', 'Greek', 'Latin', 'French', 'Old English'
    '''
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

    return fracs
    # return [e, f, n, l, g, o, u, total]


# make_pie(new_category_dict, "Proportions of etymologies in the lexicon")

def reduce_brown_pos(word, brown_tag):
    '''
    Turns a brown part of speech into a word part of speech. 
    :word: the word in question
    :brown_tag: the tag from the brown corpus
    :return: the etymdict equivalent of the brown tag, or "skip" for 
        punctuation, or None for unknowns
    '''
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
    # elif "$" in word: late addition, not reflected in submitted charts
    #     return "skip"

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
    '''
    return the etymological category of the word given
    :word: the word in question
    :brown_tag: the tag in the brown corpus
    :lemmatized: whether this word has been lemmatized yet
    :return: {'Unknown', 'Other', 'Norse', 'Greek', 'Latin', 'French', 
              'Old English'}
    '''
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
    '''
    Big function to make lots of pies. 
    Generates pies for each of six texts, with and without token frequencies, 
        and then shows them. 
    '''
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

# Take a sample of the words labeled "other"
# otherwords = []
# for words in new_category_dict.keys():
#     if new_category_dict[words] == "Other":
#         otherwords.append(words)

# print("number of 'other': ", len(otherwords))
# import random
# random.shuffle(otherwords)
# print(otherwords[:100])


# this code borrowed from stackoverflow, I'm really not sure how it works
# I only added the color. 

def plot_clustered_stacked(dfall, labels=None,
                           title="multiple stacked bar plot",  H="/", 
                           **kwargs):
    """
    Given a list of dataframes, with identical columns and index,
    create a clustered stacked bar plot. 
    labels is a list of the names of the dataframe, used for the legend
    title is a string for the title of the plot
    H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    colors = 'r', 'orange', 'b', 'c', 'm', 'y', 'g'
    colors = colors[::-1]

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      colors=colors,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    axe.add_artist(l1)
    return axe

# create dataframes

sentences = brown.tagged_sents("ca09")
word_matrix = np.matrix(make_analysis_pie(sentences, show=False)[::-1])
token_matrix = np.matrix(make_analysis_pie(sentences, show=False, token=True)[::-1])


sentences = brown.tagged_sents("cm01")
new_words = make_analysis_pie(sentences, show=False)[::-1]
new_tokens = make_analysis_pie(sentences, show=False, token=True)[::-1]
word_matrix = np.vstack((word_matrix, new_words))
token_matrix = np.vstack((token_matrix, new_tokens))


sentences = brown.tagged_sents("cp26")
new_words = make_analysis_pie(sentences, show=False)[::-1]
new_tokens = make_analysis_pie(sentences, show=False, token=True)[::-1]
word_matrix = np.vstack((word_matrix, new_words))
token_matrix = np.vstack((token_matrix, new_tokens))


sentences = brown.tagged_sents("cd07")
new_words = make_analysis_pie(sentences, show=False)[::-1]
new_tokens = make_analysis_pie(sentences, show=False, token=True)[::-1]
word_matrix = np.vstack((word_matrix, new_words))
token_matrix = np.vstack((token_matrix, new_tokens))


sentences = brown.tagged_sents("ch09")
new_words = make_analysis_pie(sentences, show=False)[::-1]
new_tokens = make_analysis_pie(sentences, show=False, token=True)[::-1]
word_matrix = np.vstack((word_matrix, new_words))
token_matrix = np.vstack((token_matrix, new_tokens))


sentences = brown.tagged_sents("cj16")
new_words = make_analysis_pie(sentences, show=False)[::-1]
new_tokens = make_analysis_pie(sentences, show=False, token=True)[::-1]
word_matrix = np.vstack((word_matrix, new_words))
token_matrix = np.vstack((token_matrix, new_tokens))



df1 = pd.DataFrame(word_matrix,
                   index=["News", "Sci-fi", "Romance", "Religion", "Legal", "Medical"],
                   columns=['Unknown', 'Other', 'Norse', 'Greek', 'Latin', 'French', 'English'][::-1])
df2 = pd.DataFrame(token_matrix,
                   index=["News", "Sci-fi", "Romance", "Religion", "Legal", "Medical"],
                   columns=['Unknown', 'Other', 'Norse', 'Greek', 'Latin', 'French', 'English'][::-1])


# Then, just call :
plot_clustered_stacked([df1, df2],["Words", "Tokens"],
                       title="Proportions of etymological origins in several texts")
plt.show()