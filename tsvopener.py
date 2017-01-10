
# import requests
import re
# import os
# import subprocess
import csv
from bs4 import BeautifulSoup


def HTMLtoUnicode(text):
    '''
    :text: text from etymonline dictionary entry
    :return: BeautifulSoup interpreted string
    '''
    text = BeautifulSoup(text)
    return text.string


def apostrophefixer(text):
    '''
    for some reason apostrophes are broken, this fixes them. 
    :text: text to be fixed
    :return: same text but with apostrophes fixed
    '''
    text = re.sub("&#039;", "\'", text)
    return text


def POSremover(text):
    '''
    do not call this function, I shouldn't have written it in the first place
    :text: some text to be cleaned of part of speech tags
    :return: text without part of speech tags
    '''
    # this is dumb. multiple keys error
    text = re.sub(" (.*)", "", text)
    return text


def writeitout(etymdict, filename):
    '''
    takes an input dict and writes it out as a tsv file. 
    :etymdict: dictionary to be written out
    :filename: the name of the file to write it to. 
    '''
    with open(filename, 'w') as writefile:
        etymwriter = csv.writer(writefile, delimiter='\t',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for key in etymdict.keys():
            word = key
            text = etymdict[key]
            etymwriter.writerow([word, text])

    # I used this to change the "see X" sections into true symlinks. using
    #    pattern match '\t[Ss]ee'
    # there were 269 of them (one, Pablum, had a capital S)


def open_tsv(filename):
    '''
    opens a tsv file and reads it into a dictionary
    :filename: the filename of the tsv file. 
    :return: a dictionary mapping the first column of the tsv to the second
    '''
    output = {}
    with open(filename, 'r') as readfile:
        reader = csv.reader(readfile, delimiter='\t', quotechar='|')

        for line in reader:
            output[line[0]] = line[1]
    return output


def makelinks(etymdict):
    '''
    fixes instances where a dictionary entry is simple "see [related word]"
    :etymdict: dictionary of words mapped to etymological entries
    :return: the same dictionary with the links resolved
    '''
    see_pattern = re.compile("^[sS]ee (.*?)[.,;]( |$)")
    for key in etymdict.keys():
        match = see_pattern.search(etymdict[key])
        if match is not None:
            linkword = match.group(1)
            print(linkword, " ", key)
            # just writing more cases until they all go through
            # some had to be done manually
            if linkword not in etymdict.keys():
                linkword = linkword + " (v.)"
            if linkword not in etymdict.keys():
                linkword = linkword[:-5] + " (n.)"
            if linkword not in etymdict.keys():
                linkword = linkword[:-5] + " (n.1)"
            if linkword not in etymdict.keys():
                linkword = linkword[:-6] + " (n.2)"
            if linkword not in etymdict.keys():
                linkword = linkword[:-6] + " (adj.)"
            if linkword not in etymdict.keys():
                linkword = linkword[:-7] + " (adv.)"
            if linkword not in etymdict.keys():
                linkword = linkword[:-7] + " (interj.)"
            if linkword not in etymdict.keys():
                linkword = linkword[:-10] + " (pron.)"
            if linkword not in etymdict.keys():
                linkword = linkword[:-8].lower()
            if linkword not in etymdict.keys():
                linkword = linkword + " (v.)"
            if linkword not in etymdict.keys():
                linkword = linkword[:-5] + " (n.)"
            if linkword not in etymdict.keys():
                linkword = linkword[:-5] + " (adj.)"
            if linkword not in etymdict.keys():
                linkword = linkword[:-7] + " (adv.)"
            if linkword not in etymdict.keys():
                linkword = linkword[:-7] + " (interj.)"
            if linkword not in etymdict.keys():
                linkword = linkword[:-10] + " (adj., adv.)"

            etymdict[key] = etymdict[linkword]

    return etymdict


etymdict = {}

# htmlremoved
with open('etymonline.tsv', 'r') as readfile:
    etymreader = csv.reader(readfile, delimiter='\t', quotechar='|')

    if __name__ == '__main__':
        for line in etymreader:
            etymdict[line[0]] = line[1]
        etymdict = makelinks(etymdict)

        writeitout(etymdict, "test1.tsv")

        # with open('temp1.tsv', 'w') as writefile:
        #   etymwriter = csv.writer(writefile, delimiter='\t', quotechar='|',
        #   quoting=csv.QUOTE_MINIMAL)
        #   for line in etymreader:
        #       #text = HTMLtoUnicode(text)
        #       #word = apostrophefixer(line[0])
        #       #text = apostrophefixer(line[1])
        #       #word = POSremover(line[0]) this is dumb
        #       text = line[1]
        #       etymwriter.writerow([word, text])

    for line in etymreader:
        etymdict[line[0]] = line[1]
