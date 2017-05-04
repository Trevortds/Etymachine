
# import requests
import re
# import os
# import subprocess
import csv
from bs4 import BeautifulSoup

languagenames = ["Greek", "Latin", "Etruscan", "Old Norse", 
                "Proto-North-Germanic", "North Germanic", "Scandanavian", 
                "Danish", "French", "Middle French", "Old French", 
                "Medieval Latin", "Middle English", "Old English", "Ingvaeonic", 
                "Frisian", "Saxon", "Proto-West-Germanic", 
                "Proto-Germanic", "Germanic", "German", "Taino", "Arawak",
                "Tupi", "Aramaic", "PIE", "Powhatan", "Italian",
                "Portuguese", "Delaware", "Mohawk", "Guarani", "Hebrew", "Sweon",
                "Sikeloi", "Nootka", "Massachuset", "Nahuatl", "Hereford",
                "Kansa", "Gypsy", "Punic", "Micmac", "Arawakan", "Lushootseed",
                "Jew", "Syriac", "Quechua", "Ojibwa", "Gussy",
                "Coelomata", "Chinese", "Athapaskan", "Algonquian", "Spanish",
                "Arabic", "Waldenses", "Amboyna", "Mahrati",
                "Seneca", "Hogarth", "Mahican", "Blackstone", "Cherokee",
                "Tewa", "Mandarin", "Mahican","Choctaw", "Narraganset",
                "Khoisan", "Narragansett", "Ojibway", "Romany", "Turk",
                "Macassar", "Mamucio", "Linnaeus", "Afrikaans", "Maniske",
                "Catawba", "Arkansas", "Herutford", "Dakota", "Romany", 
                "Abenaki", "Cree", "Sherpa", "Frankish", "Linzer", 
                "Romansch", "Walloon", "Walter", "Romansch", "Kutchin", 
                "Araucanian", "Creek", "Oneida", "Mohican", "Halkomelem", 
                "Yaqui", "Zulu", 
                ]

otherstops =    ["Shakespeare", "verbal", "the", "obsolete", "early",
                "marital", "present", "late", "figurative", "a", "earlier", 
                "experience", "noun", "common", "slang", "or", "that", "dialectical",
                ]


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
    see_pattern = re.compile("[sS]ee ([a-z]+(?: \([a-z]*?.\)| +))")
    from_pattern = re.compile("[fF]rom ([a-z]+(?: \([a-z]*?.\)| +)?)")
    plural_pattern = re.compile("(?:[pP]lural|[pP]articiple|[aA]bbreviation|[fF]orm|tense|[cC]ontraction|spelling|[Vv]ariant) of ([a-z]+(?: \([a-z]*?.\)| +)?)")
    prefix_pattern = re.compile("-? \+ ([a-z]+(?: \([a-z]*?.\)| +)?)")
    for key in etymdict.keys():
        match = see_pattern.search(etymdict[key])
        plural_match = plural_pattern.search(etymdict[key])
        from_match = from_pattern.search(etymdict[key])
        prefix_match = prefix_pattern.search(etymdict[key])

        # don't match these patterns
        for lang in languagenames:
            if lang in etymdict[key]:
                match = None
                plural_match = None
                from_match = None
                prefix_match = None
        

        if match is not None:
            linkword = match.group(1)
            # just writing more cases until they all go through
            # some had to be done manually

            if linkword.split(" ")[0] in languagenames:
                linkword = "32453459falhudszfndvwlf"
            elif linkword.split(" ")[0] in otherstops:
                linkword = "32453459falhudszfndvwlf"
                #make sure it doesn't go through. 
            else:
                print(key, " see ", linkword,)


            if linkword not in etymdict.keys():
                linkword = linkword.split(" ")[0] + " (v.)"
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
            if linkword not in etymdict.keys():
                pass
            else:
                etymdict[key] = etymdict[linkword]
                continue


        if plural_match is not None:
            linkword = plural_match.group(1)
            # just writing more cases until they all go through
            # some had to be done manually

            if linkword.split(" ")[0] in languagenames:
                linkword = "32453459falhudszfndvwlf"
            elif linkword.split(" ")[0] in otherstops:
                linkword = "32453459falhudszfndvwlf"
                #make sure it doesn't go through. 
            else:
                print(key, " form of ", linkword,)


            if linkword not in etymdict.keys():
                linkword = linkword.split(" ")[0] + " (v.)"
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
            if linkword not in etymdict.keys():
                pass
            else:
                etymdict[key] = etymdict[linkword]
                continue

        if from_match is not None:
            linkword = from_match.group(1)
            # just writing more cases until they all go through
            # some had to be done manually
            if linkword.split(" ")[0] in languagenames:
                linkword = "32453459falhudszfndvwlf"
            elif linkword.split(" ")[0] in otherstops:
                linkword = "32453459falhudszfndvwlf"
                #make sure it doesn't go through. 
            else:
                print(key, " from ", linkword,)

            if linkword not in etymdict.keys():
                linkword = linkword.split(" ")[0] + " (v.)"
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
                pass
            else:
                etymdict[key] = etymdict[linkword]
                continue

        if prefix_match is not None:
            linkword = prefix_match.group(1)
            # just writing more cases until they all go through
            # some had to be done manually
            if linkword.split(" ")[0] in languagenames:
                linkword = "32453459falhudszfndvwlf"
            elif linkword.split(" ")[0] in otherstops:
                linkword = "32453459falhudszfndvwlf"
                #make sure it doesn't go through. 
            else:
                print(key, " prefix + ", linkword,)

            if linkword not in etymdict.keys():
                linkword = linkword.split(" ")[0] + " (v.)"
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
                pass
            else:
                etymdict[key] = etymdict[linkword]
                continue

    return etymdict


etymdict = {}


if __name__ == '__main__':
    
# htmlremoved
    with open('etymonline.tsv', 'r') as readfile:
        etymreader = csv.reader(readfile, delimiter='\t', quotechar='|')

    
        for line in etymreader:
            etymdict[line[0]] = line[1]
        etymdict = makelinks(etymdict)

        writeitout(etymdict, "etymdict_no_links.tsv")

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
else:
    with open('etymdict_no_links.tsv', 'r') as readfile:
        etymreader =  csv.reader(readfile, delimiter='\t', quotechar='|')
        for line in etymreader:
            etymdict[line[0]] = line[1]
