import spacy
from spacy.gold import GoldParse
from spacy.pipeline import EntityRecognizer
import random

train_data = [
		(
            u'Jira is good for health',
            [(0, 4, u'MASALA')]
         ),
        (
            u'It is tasty, jira',
            [(14,18 , u'MASALA')]
         ),
        (
            u'People love to eat jira rice',
            [(20, 24, u'MASALA')]
         ),
        (
            u'Is it a Hindi word, Jira?',
            [(21,25, u'MASALA')]
         ),
        (
            u'Mirch is a spicy masala',
	        [(0, 5, u'MASALA')]
         ),
        (
            u'Do you like mirch?',
	        [(12,17,u'MASALA')]
        ),
	    (
	        u'I like red mirch',
	        [(12,17,u'MASALA')]
	    ),
       (
            u'There are different kinds of mirch available in India',
            [(30, 35 , u'MASALA')],
        )
	
]





entity_types = [u'MASALA']

nlp = spacy.load('en', entity=False, parser=False)
ner = EntityRecognizer(nlp.vocab, entity_types=entity_types)

for raw_text, _ in train_data:
    doc = nlp.make_doc(raw_text)
    for word in doc:
        _ = nlp.vocab[word.orth]


for itn in range(10):
    random.shuffle(train_data)
    for raw_text, entity_offsets in train_data:
        doc = nlp.make_doc(raw_text)
        #print(doc)
        gold = GoldParse(doc, entities=entity_offsets)

        #nlp.tagger(doc)
        ner.update(doc, gold)


print 'ner training done..'
while True:
	userinput = raw_input("Enter your sentence: ")
	if userinput == 'exit':
		break
	doc2 = nlp.make_doc(userinput.decode('utf-8'))
	nlp.tagger(doc2)
	ner(doc2)
	for word in doc2:
		if word.ent_type_ in entity_types:
			print(word.text, word.ent_type_)
	print

