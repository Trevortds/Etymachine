
# coding: utf-8

# # Setup

# In[7]:

import tsvopener
import pandas as pd
import numpy as np
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, vstack
from sklearn.semi_supervised import LabelPropagation, LabelSpreading




regex_categorized = tsvopener.open_tsv("categorized.tsv")
human_categorized = tsvopener.open_tsv("human_categorized.tsv")

# Accuracy Check
#
# match = 0
# no_match = 0
# for key in human_categorized:
#     if human_categorized[key] == regex_categorized[key]:
#         match += 1
#     else:
#         no_match += 1
# 
# print("accuracy of regex data in {} human-categorized words".format(
#             len(human_categorized)))
# print(match/(match+no_match))
# 
# accuracy of regex data in 350 human-categorized words
# 0.7857142857142857


# # Prepare Vectors

# In[8]:

# set up targets for the human-categorized data
targets = pd.DataFrame.from_dict(human_categorized, 'index')
targets[0] = pd.Categorical(targets[0])
targets['code'] = targets[0].cat.codes
# form: | word (label) | language | code (1-5)

tmp_dict = {}
for key in human_categorized:
    tmp_dict[key] = tsvopener.etymdict[key]
supervised_sents = pd.DataFrame.from_dict(tmp_dict, 'index')

all_sents = pd.DataFrame.from_dict(tsvopener.etymdict, 'index')
vectorizer = CountVectorizer(stop_words='english', max_features=10000)
all_sents.index.get_loc("anyways (adv.)")


# In[9]:

# vectorize the unsupervised vectors.

vectors = vectorizer.fit_transform(all_sents.values[:,0])

print(vectors.shape)
# supervised_vectors = vectorizer.fit_transform(supervised_data.values[:,0])


# In[10]:

# add labels 

# initialize to -1
all_sents['code'] = -1


supervised_vectors = csr_matrix((len(human_categorized),
                                 vectors.shape[1]), 
                                dtype=vectors.dtype)

j = 0
for key in supervised_sents.index:
    all_sents.loc[key]['code'] = targets.loc[key]['code']
    i = all_sents.index.get_loc(key)
    supervised_vectors[j] = vectors[i]
    j += 1


    
# supervised_vectors = csr_matrix((len(human_categorized),
#                                  unsupervised_vectors.shape[1]), 
#                                 dtype=unsupervised_vectors.dtype)

# j = 0
# for key in supervised_data.index:
#     i = unsupervised_data.index.get_loc(key)
#     supervised_vectors[j] = unsupervised_vectors[i]
#     j += 1


    
all_sents.loc['dicky (n.)']


# In[ ]:




# # Use Scikit's semisupervised learning
# 
# There are two semisupervised methods that scikit has. Label Propagation and Label Spreading. The difference is in how they regularize. 

# In[23]:

num_points = 1000
num_test = 50

x = vstack([vectors[:num_points], supervised_vectors]).toarray()
t = all_sents['code'][:num_points].append(targets['code'])

x_test = x[-num_test:]
t_test = t[-num_test:]
x = x[:-num_test]
t = t[:-num_test]

label_prop_model = LabelSpreading(kernel='knn')
from time import time

print("fitting model")
timer_start = time()
label_prop_model.fit(x, t)
print("runtime: %0.3fs" % (time()-timer_start))


# In[24]:

print("done!")

# unsupervised_data['code'].iloc[:1000]


# In[11]:

import pickle 

# with open("classifiers/labelspreading_knn_all_but_100.pkl", 'bw') as writefile:
#     pickle.dump(label_prop_model, writefile)


# In[25]:


import smtplib
 
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login("trevortds3@gmail.com", "Picardy3")
 
msg = "Job's done!"
server.sendmail("trevortds3@gmail.com", "trevortds@gmail.com", msg)
server.quit()


# In[15]:

targets


# # Measuring effectiveness. 
# 
# 

# In[26]:

from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score



t_pred = label_prop_model.predict(x_test)

print("Metrics based on 50 hold-out points")

print("Macro")
print("accuracy: %f" % accuracy_score(t_test, t_pred))
print("precision: %f" % precision_score(t_test, t_pred, average='macro'))
print("recall: %f" % recall_score(t_test, t_pred, average='macro'))
print("f1: %f" % f1_score(t_test, t_pred, average='macro'))
print("\n\nMicro")
print("accuracy: %f" % accuracy_score(t_test, t_pred))
print("precision: %f" % precision_score(t_test, t_pred, average='micro'))
print("recall: %f" % recall_score(t_test, t_pred, average='micro'))
print("f1: %f" % f1_score(t_test, t_pred, average='micro'))

from sklearn import metrics
import matplotlib.pyplot as pl

labels = ["English", "French", "Greek", "Latin","Norse", "Other"]
labels_digits = [0, 1, 2, 3, 4, 5]
cm = metrics.confusion_matrix(t_test, t_pred, labels_digits)

fig = pl.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
pl.title("Label Spreading with KNN kernel (k=7)")
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
pl.xlabel('Predicted')
pl.ylabel('True')

pl.show()


# # PCA: Let's see what it looks like
# 
# Performing PCA

# In[11]:

supervised_vectors


# In[13]:

import matplotlib.pyplot as pl

u, s, v = np.linalg.svd(supervised_vectors.toarray())
pca = np.dot(u[:,0:2], np.diag(s[0:2]))



# In[15]:

english = np.empty((0,2))
french = np.empty((0,2))
greek = np.empty((0,2))
latin = np.empty((0,2))
norse = np.empty((0,2))
other = np.empty((0,2))

for i in range(pca.shape[0]):
    if targets[0].iloc[i] == "English":
        english = np.vstack((english, pca[i]))
    elif targets[0].iloc[i] == "French":
        french = np.vstack((french, pca[i]))
    elif targets[0].iloc[i] == "Greek":
        greek = np.vstack((greek, pca[i]))
    elif targets[0].iloc[i] == "Latin":
        latin = np.vstack((latin, pca[i]))
    elif targets[0].iloc[i] == "Norse":
        norse = np.vstack((norse, pca[i]))
    elif targets[0].iloc[i] == "Other":
        other = np.vstack((other, pca[i]))
        
pl.plot( english[:,0], english[:,1], "ro", 
          french[:,0],  french[:,1], "bs",
           greek[:,0],   greek[:,1], "g+",
           latin[:,0],   latin[:,1], "c^",
           norse[:,0],   norse[:,1], "mD",
           other[:,0],   other[:,1], "kx")
pl.axis([-5,0,-2, 5])
pl.show()


# In[17]:

print (s)

