import tsvopener

etymdict = tsvopener.etymdict


category_dict = {}

hierarchies = {
    "Greek": ["Greek"],
    "Latin": ["Latin", "Etruscan"],
    "Norse": ["Old Norse", "Proto-North-Germanic", "North Germanic",
              ],
    "French": ["French", "Middle French", "Old French", "Medieval Latin",
               ],
    "English": ["Old English", "Ingvaeonic", "Frisian", "Saxon",
                "Proto-West-Germanic", "Proto-Germanic", "Germanic"],
}


def regex_categorize(definition):
    '''
    Steps through the above hierarchies to find which categorization is
        appropriate based on what words are present. The older languages come
        first because they may be overwritten later.

    '''
    output = ""
    for category, chain in hierarchies.items():
        for item in chain:
            if item in definition:
                output = category

    if output == "":
        output = "Other"
    return output


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
        cat = regex_categorize(definition)
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
