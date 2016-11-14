
import requests
import re
import os
import subprocess
import csv


with open('etymbackup.tsv', 'r') as readfile:
    etymreader = csv.reader(readfile, delimiter='\t', quotechar='|')
    with open('etymonline.tsv', 'w') as writefile:
        etymwriter = csv.writer(writefile, delimiter='\t',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for line in etymreader:
            output = re.sub("\n", " ", line[1])
            etymwriter.writerow(line[0], output)
