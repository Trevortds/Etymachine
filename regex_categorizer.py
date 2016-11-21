import tsvopener
import random
import re

etymdict = tsvopener.etymdict

danger = False

category_dict = {"buzz (n.)": "Other"}

hierarchies = {
    "Greek": ["Greek"],
    "Latin": ["Latin", "Etruscan"],
    "Norse": ["Old Norse", "Proto-North-Germanic", "North Germanic",
              "Scandanavian", "Danish"],
    "French": ["French", "Middle French", "Old French", "Medieval Latin",
               ],
    "English": ["Old English", "Ingvaeonic", "Frisian", "Saxon",
                "Proto-West-Germanic", "Proto-Germanic", "Germanic"],
}

order = ["Greek", "Latin", "Norse", "French", "English"]

from_pattern = re.compile("[Ff]rom (\w+ \(\w+.\d?\)|\w*-)")
from_paren_optional_pattern = re.compile("[Ff]rom (\w\w+( \(\w+.\d?\))?|\w*-)")
see_pattern = re.compile("[sS]ee (\w+ \(\w+.\d?\))")
see_paren_optional_pattern = re.compile("[Ss]ee (\w+( \(\w+.\d?\))?)")
variant_pattern = re.compile("[Vv]ariant of (\w+ \(\w+.\d?\))")


def regex_categorize(word, definition, depth=0):
    '''
    Steps through the above hierarchies to find which categorization is
        appropriate based on what words are present. The older languages come
        first because they may be overwritten later.

    '''
    # making this recursive, so we need a base case
    if word in category_dict.keys():
        return category_dict[word]
    if depth >= 10:
        return "ERR"

    output = ""
    for category in order:
        chain = hierarchies[category]
        for item in chain:
            if item in definition:
                output = category

    if output == "":
        # there's no language mentioned: doesn't mean other automaticall
        # could be a compound
        # check if there's a from x
        done = False
        if from_pattern.search(definition) is not None:
            # check if x is in dictionary, if so, use the category of x
            tst = from_pattern.search(definition).group(1)
            if tst in etymdict.keys():
                output = regex_categorize(tst, etymdict[tst], depth+1)
            if output != "ERR":
                done = True
        # check if there's a see x
        if not done and see_pattern.search(definition) is not None:
            # check is x is in dictionary, if so, use the category of x
            tst = see_pattern.search(definition).group(1)
            if tst in etymdict.keys():
                output = regex_categorize(tst, etymdict[tst], depth+1)
            if output != "ERR":
                done = True
        # variant of x
        if not done and variant_pattern.search(definition) is not None:
            tst = variant_pattern.search(definition).group(1)
            if tst in etymdict.keys():
                output = regex_categorize(tst, etymdict[tst], depth+1)
            if output != "ERR":
                done = True
        # danger zone: these patterns will match non-etymological connections,
        #     like "from portuguese"
        if (not done and danger and
                (from_paren_optional_pattern.search(definition) is not None)):
            tst = from_paren_optional_pattern.search(definition).group(1)
            if tst in etymdict.keys():
                output = regex_categorize(tst, etymdict[tst], depth+1)
            if output != "ERR":
                done = True
        if (not done and danger and
                (see_paren_optional_pattern.search(definition) is not None)):
            tst = see_paren_optional_pattern.search(definition).group(1)
            if tst in etymdict.keys():
                output = regex_categorize(tst, etymdict[tst], depth+1)
            if output != "ERR":
                done = True
        if not done:
            # else, other
            output = "Other"
    return output


def manually_check(etymdict, category_dict, num):
    '''
    takes judgements from the user, returns number right, number wrong,
        number total
    '''
    wordlist = list(etymdict.keys())
    random.shuffle(wordlist)
    # shuffle and then truncate the list
    wordlist = wordlist[:num]
    numright = 0
    numwrong = 0
    numtotal = 0
    for word in wordlist:
        print("Word: ", word, " Category: ", category_dict[word])
        print("Definition: ", etymdict[word])
        numtotal += 1
        while True:
            judge = input("\nCorrect? [enter for yes, n for no, c to close]:")
            if judge == '':
                numright += 1
            elif judge == 'n':
                numwrong += 1
            elif judge == 'c':
                return numright, numwrong, numtotal
            else:
                continue
            break


if __name__ == '__main__':
    gr = 0
    lat = 0
    nor = 0
    fr = 0
    eng = 0
    other = 0
    tot = 0

    print("categorizing")

    for word, definition in etymdict.items():
        cat = regex_categorize(word, definition)
        category_dict[word] = cat
        if cat == "English":
            eng += 1
        elif cat == "French":
            fr += 1
        elif cat == "Greek":
            gr += 1
        elif cat == "Latin":
            lat += 1
        elif cat == "Norse":
            nor += 1
        elif cat == "Other":
            other += 1
        tot += 1

    print("done")

    print("English: ", eng, " percent of total: ", eng / tot * 100)
    print("French: ", fr, " percent of total: ", fr / tot * 100)
    print("Norse: ", nor, " percent of total: ", nor / tot * 100)
    print("Latin: ", lat, " percent of total: ", lat / tot * 100)
    print("Greek: ", gr, " percent of total: ", gr / tot * 100)
    print("Other: ", other, " percent of total: ", other / tot * 100)

    print("Total: ", tot)

    tsvopener.writeitout(category_dict, "categorized.tsv")
