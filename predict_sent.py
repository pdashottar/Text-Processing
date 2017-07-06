from keras.models import load_model
# returns a compiled model
# identical to the previous one
model = load_model('lstm_model.h5')

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

from spacy.en import English
nlp = English()


# In[8]:

import numpy as np
def build_input_data_nlp(sentences, labels):
    # uses spacy.en and English()
    
    nb_sentences = len(sentences)          # no. of sentences        
    nb_tokens = len(sentences[0])          # = 50 in this case
    delist = sentences[0]
    word_vec_dim = nlp(delist[0].decode('utf8'))[0].vector.shape[0]       #taken 1st word of 1st sentence
    # .vector is a 1-dimensional numpy array of 32-bit floats. The default English model installs vectors for one million vocabulary entries, using the 300-dimensional vectors
    # shape attribute for numpy arrays returns the dimensions of the array. If Y has n rows and m columns, then Y.shape is (n,m). So Y.shape[0] is n
    sentences_matrix = np.zeros((nb_sentences, nb_tokens, word_vec_dim))       # np.zero() Return a new array of given shape and type, filled with zeros
    #print sentences_matrix.shape
    for k in xrange(nb_sentences):
        delist = sentences[k]
        m = len(delist)
        for i in xrange(len(delist)):      # xrange is faster compare to range
            tokens = nlp(delist[i].decode('utf8'))       # token = each word (in loop)
            for j in xrange(len(tokens)):
                sentences_matrix[k, i, :] = tokens[j].vector
    ss = sentences_matrix.shape
    y = np.array(labels)
    x = sentences_matrix
    return [x,y]
    
while True:
	userinput = raw_input("Enter your sentence: ")
	if userinput == 'exit':
	    break
	doc2 = clean_str(userinput)
	new_doc = [doc2.split(' ')]
	len(new_doc[0])

	seq_len = 80
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


	# In[45]:

	prediction = model.predict(X,verbose=1)
	print prediction