
# coding: utf-8

# In[5]:

import argparse

parser = argparse.ArgumentParser()
#parser.add_argument("--filter_sizes",nargs = '?', const = 1, default = (2,3,4))
parser.add_argument("--pool_dropout", nargs = '?', const = 1, default = 0.8, type = float)
parser.add_argument("--dense_dropout", nargs = '?', const = 1, default = 0.8, type = float)
parser.add_argument("--epochs", nargs = '?', const = 1, default = 4, type = int)
parser.add_argument("--batch_size", nargs = '?', const = 1, default = 128, type = int)
parser.add_argument("--num_filters", nargs = '?', const = 1, default = 5, type = int )
parser.add_argument("--strides", nargs = '?', const = 1, default = 1, type = int )
args = parser.parse_args()


#print "filter size (default) = %d \n" %args.epochs
# In[20]:


import random
lines = []
with open("C:\Users\pooja\Desktop\intern\IMDB-data.txt", "r") as file:
    #data = f.read().split('\n')
    for line in file:   # reads line by line
        line = line.strip() # to remove \n
        lines.append(line)
random.shuffle(lines)


# In[21]:

train_data = lines[:6000]
test_data = lines[6000:]

thefile = open('C:\Users\pooja\Desktop\intern\IMDB-test-data.txt', 'w')

for item in test_data:
    thefile.write("%s\n" % item)
lines = train_data

#lines = []     # declared a dictionary
#with open("C:\Users\pooja\Desktop\intern\IMDB-data.txt") as file:  #by using with it also closes the file after use
#    for line in file:   # reads line by line
#        line = line.strip() # to remove \n
#        lines.append(line)
#lines[:10]


# In[22]:

Y = [lines[y].split("\t")[0] for y in range(len(lines))]

X = [lines[y].split("\t")[1] for y in range(len(lines))]


# In[23]:

X[:10]


# In[24]:

import re
def clean_str(string):

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# In[25]:

new_X = [clean_str(s) for s in X]
new_X[:5]


# In[26]:

new_X = [s.split(' ') for s in new_X]         # new_X is a list of list
#print new_X[1]


# In[27]:

# counting max size of a line
max_len = 0
for itn in range(len(Y)):
    if max_len < len(new_X[itn]) :
        max_len = len(new_X[itn])
print max_len


# In[28]:

seq_len = max_len
padding_word = '<PAD/>'
padded_sentences = []
for i in range(len(new_X)):
    sentence = new_X[i]
    num_padding = seq_len - len(sentence)
    new_sentence = sentence + [padding_word] * num_padding
    padded_sentences.append(new_sentence)

#padded_sentences[:3]


# In[29]:

from spacy.en import English
nlp = English()


# In[30]:

import numpy as np
def build_input_data_nlp(sentences, labels):
    # uses spacy.en and English()
    
    nb_sentences = len(sentences)          # no. of sentences        
    nb_tokens = len(sentences[0])          # = 50 in this case
    delist = sentences[0]
    word_vec_dim = nlp(delist[0].decode('utf8'))[0].vector.shape[0]       #taken 1st word of 1st sentence
    # .vector is a 1-dimensional numpy array of 32-bit floats. The default English model installs vectors for one million vocabulary entries, using the 300-dimensional vectors
    # shape attribute for numpy arrays returns the dimensions of the array. If Y has n rows and m columns, then Y.shape is (n,m). So Y.shape[0] is n
    #print word_vec_dim
    sentences_matrix = np.zeros((nb_sentences, nb_tokens, word_vec_dim))       # np.zero() Return a new array of given shape and type, filled with zeros
    # print sentences_matrix.shape
    for k in xrange(nb_sentences):
        delist = sentences[k]
        m = len(delist)
        for i in xrange(len(delist)):      # xrange is faster compare to range
            tokens = nlp(delist[i].decode('utf8'))       # token = each word (in loop)
            for j in xrange(len(tokens)):
                sentences_matrix[k, i, :] += tokens[j].vector
    ss = sentences_matrix.shape
    #print ss
    y = np.array(labels)
    x = sentences_matrix
    return [x,y]


# In[31]:

X,Y = build_input_data_nlp(padded_sentences, Y)

#X[0][0][0]

len(Y)              # our IMDB-train-data size = 6000 lines
                    # test-data size = 1086 lines 

# since our data file has 2 categories which are sequentially given, before dividing it into train & test data we need to shuffle it
#shuffle_indices = np.random.permutation(np.arange(len(Y)))      # len(Y) gives no. of sentences and this fun randomly generates indices in the given range
#X_shuffled = X[shuffle_indices]
#Y_shuffled1 = Y[shuffle_indices].argmax(axis=1)
#Y_shuffled = Y[shuffle_indices]

#Y_shuffled[:10]


# In[32]:

#!pip install git+git://github.com/Theano/Theano.git --upgrade --no-deps
#!pip install git+git://github.com/fchollet/keras.git --upgrade --no-deps


# In[33]:

from keras.layers import Input, Dense, Conv1D, MaxPooling1D , Flatten , Dropout
from keras.models import Model
from keras.layers.merge import concatenate


# In[34]:

# we've to define i/p for our model, seq_len =50 (no. of words in a sentence). 300 is dimention of vecotor corresp. to a word
reviews = Input(shape = (seq_len , 300))              # this returns a tensor


# In[35]:


convs = []
filter_sizes = (2,3,4,5,6)     # 3X4 filter (kernel)
args.num_filters = 5
#dropout_probability = (0.5,0.6)
for fsz in filter_sizes:
	conv = Conv1D(filters=args.num_filters,
	                     kernel_size=fsz,
	                     padding='valid',      # valid means no padding
	                     activation='relu',
	                     strides=args.strides,
	                     )(reviews)

	pool = MaxPooling1D(pool_size=2, strides=args.strides)(conv)      # pool_size is size of max-pooling window
	pool = Dropout(args.pool_dropout)(pool)
	flatten = Flatten()(pool)
	convs.append(flatten)

if len(filter_sizes)>1:
    out = concatenate(convs)
else:
 	out = convs[0]


# In[36]:

# a layer instance is callable on a tensor, and returns a tensor
Dense_out = Dense(128, activation = 'relu')(out)   
# Dense(128) is a fully-connected layer with 128 hidden units.
Dense_out = Dropout(args.dense_dropout)(Dense_out)

Dense_out = Dense(64, activation = 'relu')(Dense_out)   
# Dense(64) is a fully-connected layer with 64 hidden units.
Dense_out = Dropout(args.dense_dropout)(Dense_out)


# In[37]:

Dense_out = Dense(1, activation = 'sigmoid')(Dense_out)          # 1 = final classification
#Dropout(0.5)


# In[38]:

my_model = Model(inputs = reviews, outputs = Dense_out)              # This creates a model that includes the Input layer and two Dense layers


# In[39]:

#!pip install setuptools==33.1.1        # all this for plot_model() to run
#!pip install pyparsing==1.5.7
#!pip install pydot==1.0.28

import pydot
import graphviz
from keras.utils.vis_utils import plot_model
my_model.compile(loss='binary_crossentropy', # using the cross-entropy loss function
              optimizer='rmsprop', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy


plot_model(my_model, to_file='model_reviews.png', show_shapes=True)


# In[40]:

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y , test_size=0.25,random_state=42)
#X_train[:5]
Y_train[:10]


# In[41]:

my_model.fit(X_train, Y_train, epochs=args.epochs, batch_size = args.batch_size, shuffle = True, validation_split=0.2, verbose = 1)               # shuffle- for shuffling data - it gives better results


# In[42]:

score = my_model.evaluate(X_test, Y_test, verbose=1)               # Returns the loss value & metrics values for the model in test mode.


# In[43]:

my_model.save('my_model.h5')         # creates a HDF5 file 'my_model.h5'

# In[44]:

'''while True:
    userinput = raw_input("Enter your sentence: ")
    if userinput == 'exit':
        break
    doc2 = clean_str(userinput)

new_doc = [doc2.split(' ')]
len(new_doc[0])

seq_len = 50
padding_word = '<PAD/>'
#padded_sentence = []

sentence = new_doc[0]
num_padding = seq_len - len(sentence)
new_sentence = sentence + [padding_word] * num_padding
   # padded_sentence.append(new_sentence)

#padded_sentences

labels = [0] 
X,Y = build_input_data_nlp([new_sentence], labels) 

len(new_sentence)
'''


# In[45]:

'''prediction = my_model.predict(X,verbose=1)'''


# In[ ]:



