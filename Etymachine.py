import requests
import re
import os
import subprocess
import csv
import nltk

test_text = "four score and seven years ago, our fathers brought fourth on this continent a new nation."
tokenized_text = nltk.word_tokenize(test_text)


with open('etymwn.tsv', 'rb') as csvfile:
	print "blah1"
	etymreader = csv.reader(csvfile, delimiter='\t')
	print "blah2"


	print etymreader.next()
	# eng: $word

	'''
	comment

	'''


	# for row in etymreader:
		
	# 	print ", ".join(row)
	# 	print "hi"
	
