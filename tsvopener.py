
# import requests
import re
# import os
# import subprocess
import csv
from bs4 import BeautifulSoup


def HTMLtoUnicode(text):
    text = BeautifulSoup(text)
    return text.string


def apostrophefixer(text):
    text = re.sub("&#039;", "\'", text)
    return text


def POSremover(text):
    # this is dumb. multiple keys error
    text = re.sub(" (.*)", "", text)
    return text


def writeitout(etymdict, filename):
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
    output = {}
    with open(filename, 'r') as readfile:
        reader = csv.reader(readfile, delimiter='\t', quotechar='|')

        for line in reader:
            output[line[0]] = line[1]
    return output


def makelinks(etymdict):
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
