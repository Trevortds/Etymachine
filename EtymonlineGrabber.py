#!/usr/bin/env python

import requests
import re
import os
import subprocess
import csv

alphabet = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

num_pages = {'a':59, 'b':52, 'c':83, 'd':45, 'e':37, 'f':41, 'g':34, 'h':36, 'i':36, 'j':9, 'k':8, 
	'm':55, 'n':19, 'o':21, 'p':80, 'q':5, 'r':45, 's':107, 't':45, 'u':23, 'v':14, 'w':21, 'x':0, 'y':3, 'z':2}

word_regex = re.compile('allowed_in_frame=0\">([a-zA-Z\(\)0-9 -\.]*)</a> <a href=') # grabs the title of each entry from html
etym_regex = re.compile('<dd.*>.*</dd>')
html_purge_regex = re.compile('>(.*?)<')
from_regex = re.compile('from (([A-Z][a-zA-Z-]*) )+')



with open('etymologies.csv', 'wb') as csvfile:
	etymwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	for letter in alphabet:
		for i in range(0, num_pages[letter]):
			r = requests.get("http://www.etymonline.com/index.php?l=" + letter + "&p=" + str(i) + "&allowed_in_frame=0")
			webpage = r.text;
			list_of_words = re.findall(word_regex, webpage)
			list_of_etyms = re.findall(etym_regex, webpage)

			for j in range(0, len(list_of_words)):
				no_html_list = re.findall(html_purge_regex, list_of_etyms[j])
				no_html_string = "".join(no_html_list)

				#INSERT FANCY NLP STUFF HERE
				lang = re.findall(from_regex, no_html_string)
				
				print lang
				lang_2 = [lang[i][1] for i in range(0,len(lang))] #taking the second row of lang, which is a multidimensnional array

				
				
				
				print lang_2;

				if (len(lang_2) == 0):
					save = subprocess.check_output("python GrabberGui.py \'" + list_of_words[j] + "\' \'" +
						no_html_string + "\'", shell=True)
				elif (lang_2[0] == "Latin"):
					save = "Latin"
				elif (lang_2[0] == "Greek"):
					save = "Greek"
				elif (lang_2[0] == "Old"):
					print lang_2
					save = ''.join(lang_2)
				else:
					save = subprocess.check_output("python GrabberGui.py \'" + list_of_words[j] + "\' \'" +
						no_html_string + "\'", shell=True)


				'''
				list(set(lang))

				lang_2 = list(lang)
				#make this it's own function
				for word in lang_2:
					if (word == "Latin" or word=="Greek" or word=="Old" or word=="late"):
						continue
					else:
						lang.remove(word)
				

				if(len(lang) > 1 or len(lang) == 0):
					#print lang
					save = subprocess.check_output("python GrabberGui.py \'" + list_of_words[j] + "\' \'"+
						no_html_string + "\'", shell=True)
				elif (lang[0] == "Latin"):
					save = "Latin"
				elif (lang[0] == "Greek"):
					save = "Greek"
				else:
					save = subprocess.check_output("python GrabberGui.py \'" + list_of_words[j] + "\' \'" +
						no_html_string + "\'", shell=True)

				'''

				if not (save == "skip"):
					etymwriter.writerow([list_of_words[j], save])









'''
next steps:
CHECK make a gui
CHECKiterate through all the webpages, when there is a collision, call gui
CHECKchange gui to include the word 

implement an option to not include a word in the corpus, ie, when it returns -1, don't execute the etymwriter line


'''