from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import csv
import json
import collections
from vutils import *

data_dict = collections.defaultdict()
with open('bay_nv_data.csv') as f:
	reader = csv.DictReader(f)
	for row_no, row in enumerate(reader):
		if row_no > 3000:
			continue
			
		k = row['expected semantic name']
		v = row['correct_segment_v3']
		data_dict[k] = data_dict.get(k, [])
		data_dict[k].append(v)
#
y_labels = []
corpus_tok = []
corpus = []

se_n_gram = 2

start_words = []
end_words = []
body_words = []
_vs = []
for k in data_dict.keys():
	y_labels.append(k)
	vs = data_dict[k]
	dout = ' EOS '.join(vs)
	dout = dout.replace('  ', ' ')
	corpus.append(dout)
	corpus_tok.append(dout.split(' '))

	for v in vs:
		_v = v.split(' ')
		_vs.append(_v)

import random
random.seed(0)
random.shuffle(_vs)
for _v in _vs[:int(len(_vs)*3/4)]:
	if len(_v) >= 2:
		start_words.append(_v[:se_n_gram])
		end_words.append(_v[-1*se_n_gram:])
		body_words.append(_v)


start_end_words = []
start_end_words.extend(start_words)
start_end_words.extend(end_words)

for ew in end_words:
	for sw in start_words:
		tmp = []
		tmp.append(ew[-1])
		tmp.append(sw[0])
		start_end_words.append(tmp)

print('start calcualte se_word prob.')
start_end_word_cnt_dict = collections.defaultdict()
for _start_end_word in start_end_words:
	#print(_start_end_word)
	start_end_word = ' '.join(_start_end_word)
	start_end_word_cnt_dict[start_end_word] = start_end_word_cnt_dict.get(start_end_word, 0)
	start_end_word_cnt_dict[start_end_word] += 1

	se_word_split = start_end_word.split(' ')
	se_word_split_length = len(se_word_split)
	if se_word_split_length > 1:
		for se_word in se_word_split:
			start_end_word_cnt_dict[se_word] = start_end_word_cnt_dict.get(se_word, 0)
			start_end_word_cnt_dict[se_word] += 1

se_word_prob = []
body_words_text = ' | '.join([ ' '.join(_) for _ in body_words ])
for se_word in start_end_word_cnt_dict.keys():
	se_count = start_end_word_cnt_dict[se_word]
	bd_count = body_words_text.count(se_word)

	n_support = (se_count+bd_count)

	se_prob = se_count/n_support


	se_word_prob.append([se_word, se_prob, n_support])
	print(se_word)
	print(se_prob)
	print('-'*3)

se_word_prob = sorted(se_word_prob, key = lambda x : x[1])
#print(se_word_prob[-1*int(len(se_word_prob)*3/4):])
se_word_prob = [ _ for _ in se_word_prob if _[1] > 0.85 ]

vectorizer = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 4) , tokenizer = lambda x : x.split(' '))
vectorizer.fit(corpus)
vocab = vectorizer.get_feature_names()

tw_dict = collections.defaultdict()

eos_idx = vocab.index('eos')
for c_no, c in enumerate(corpus):
	y_label = y_labels[c_no]
	Y = [ list(y) for y in vectorizer.transform([c]).toarray() ][0] # tf-idf

	tmp_tfidf = list(zip(vocab, Y))
	tmp_tfidf = [ t for t in tmp_tfidf if t[1] > Y[eos_idx] ]

	#print(y_labels[c_no])
	#print(tmp_tfidf)

	for w, p in tmp_tfidf:
		tmp_u = zero_vec(len(y_labels))
		tmp_u[y_labels.index(y_label)] = p
		tw_dict[w] = tw_dict.get(w, zero_vec(len(y_labels)))
		tw_dict[w] = merge_vec(tw_dict[w], tmp_u)

data_model = { 'se_word_prob': se_word_prob, 'tw_dict' : tw_dict , 'y_labels' : y_labels , '__name' : 'TermWeight' , '__version' : '0.1' }
with open('tw_model.json', 'w') as f:
	f.write(json.dumps(data_model))
	f.flush()
	f.close()

for w in tw_dict.keys():
	p_dist = tw_dict[w]
	print(w)
	for y_idx, p in enumerate(p_dist):
		if p > 0:
			print('    {} : {}'.format(str(y_labels[y_idx]), str(p)))

	print()








