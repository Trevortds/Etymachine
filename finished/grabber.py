
import requests
import re
import os
import subprocess
import csv

alphabet = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

alphabet = ["a"] #for debugging

num_pages = {'a':59, 'b':52, 'c':83, 'd':45, 'e':37, 'f':41, 'g':34, 'h':36, 'i':36, 'j':9, 'k':8, 
	'm':55, 'n':19, 'o':21, 'p':80, 'q':5, 'r':45, 's':107, 't':45, 'u':23, 'v':14, 'w':21, 'x':0, 'y':3, 'z':2}

num_pages = {'a':5} #for debugging

word_regex = re.compile('allowed_in_frame=0\">([a-zA-Z\(\)0-9 / -\.\&\;]*)</a> <a href=') # grabs the title of each entry from html
word_regex = re.compile('allowed_in_frame=0\">(.*)</a> <a href=') # grabs the title of each entry from html
etym_regex = re.compile('(<dd.*?>(?:.|\n)*?</dd>)')
html_purge_regex = re.compile('>(.*?)<')
from_regex = re.compile('from (([A-Z][a-zA-Z-]*) )+')

with open('etymonline.tsv', 'w') as csvfile:
	etymwriter = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	for letter in alphabet:
		for i in range(0, num_pages[letter]):
			r = requests.get("http://www.etymonline.com/index.php?l=" + letter + "&p=" + str(i) + "&allowed_in_frame=0")
			webpage = r.text;
			list_of_words = re.findall(word_regex, webpage)
			list_of_etyms = re.findall(etym_regex, webpage)
			#print(list_of_etyms)
			print(list_of_words)
			if len(list_of_etyms) != len(list_of_words):
				print(list_of_words)
				raise Exception("definition/wordlist mismatch at "+ letter+ str(i)+
					": \n "+ str(len(list_of_words))+ " words "+ str(len(list_of_etyms))+" definitions")

			for j in range(0, len(list_of_etyms)):

				etymwriter.writerow([list_of_words[j], list_of_etyms[j]])