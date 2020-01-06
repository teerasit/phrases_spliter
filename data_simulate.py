import json
import csv
import collections
from vutils import *
import math
import os
import glob

with open('tw_model.json') as f:
	raw = f.read()
	tw_model = json.loads(raw)
	y_labels = tw_model['y_labels']
	tw_dict = tw_model['tw_dict']
	se_word_prob = tw_model['se_word_prob']

stopwords = [ _[0] for _ in se_word_prob ]
stopword_probs = collections.defaultdict()
for se in se_word_prob:
	stopword_probs[se[0]] = se[1]*se[2]

def get_prob_vec(w, scale = 1):
	if '#' in w.split(' '):
		return zero_vec(len(y_labels))

	if 'eos' in w.split(' '):
		return zero_vec(len(y_labels))

	if w in tw_dict.keys():
		vec = tw_dict[w]
	else:
		vec = zero_vec(len(y_labels))

	for _ in range(len(vec)):
		vec[_] = vec[_] * scale

	return vec

def transform(tw_model, ss):
	tmp_zero = zero_vec(len(y_labels))
	tmp_us = []
	w_len = len(ss)
	for w_no, w in enumerate(ss):

		uni_vec = get_prob_vec(w, scale = 1)

		bi_vecs = []
		if w_no - 1 >= 0:
			bi_vec = get_prob_vec(ss[w_no-1]+' '+ss[w_no], scale = 0.75)
			bi_vecs.append(bi_vec)
		if w_no + 1 < w_len:
			bi_vec = get_prob_vec(ss[w_no]+' '+ss[w_no+1], scale = 0.75)
			bi_vecs.append(bi_vec)

		tri_vecs = []
		if w_no - 2 >= 0:
			tri_vec = get_prob_vec(ss[w_no-2]+' '+ss[w_no-1]+' '+ss[w_no], scale = 0.5)
			tri_vecs.append(tri_vec)
		if w_no - 1 >= 0 and w_no + 1 < w_len:
			tri_vec = get_prob_vec(ss[w_no-1]+' '+ss[w_no]+' '+ss[w_no+1], scale = 0.5)
			tri_vecs.append(tri_vec)
		if w_no + 2 < w_len:
			tri_vec = get_prob_vec(ss[w_no]+' '+ss[w_no+1]+' '+ss[w_no+2], scale = 0.5)
			tri_vecs.append(tri_vec)

		n_vecs = []
		n_vecs.extend(bi_vecs)
		n_vecs.extend(tri_vecs)

		vec = uni_vec
		for n_vec in n_vecs:
			vec = sum_vec(vec, n_vec)

		tmp_us.append(vec)
	return tmp_us

def merge_tmp_us(tmp_us):
	tmp_u = zero_vec(len(y_labels))
	for tmp_uu in tmp_us:
		tmp_u = merge_vec(tmp_uu, tmp_u)
	return tmp_u

data_dict = collections.defaultdict()
with open('bay_nv_data.csv') as f:
	reader = csv.DictReader(f)
	for row_no, row in enumerate(reader):
		if row_no <= 3000:
			continue

		k = row['expected semantic name']
		v = row['correct_segment_v3']
		data_dict[k] = data_dict.get(k, [])
		data_dict[k].append(v)


from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)

import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.patches as patches
import rangeset

# create a color palette
palette = plt.get_cmap('tab20')
palette = plt.get_cmap('tab10')

def draw_rect(plt, idx, width = 1):
	ax = plt.gca()
	rect = patches.Rectangle((idx-0.5,0),width,2+width/2,linewidth=1,edgecolor='r',facecolor='none', color= '#a0ffff' if width <= 1 else '#c0ffff')
	ax.add_patch(rect)


def draw_rect_cut(plt, idx, width = 0.1):
	ax = plt.gca()
	rect = patches.Rectangle((idx + 0.5,0),width,2+width/2,linewidth=1,edgecolor='r',facecolor='none', color= '#a0ffff' if width <= 1 else '#c0ffff')
	ax.add_patch(rect)


def draw_centroid(plt, idx, color):
	ax = plt.gca()
	sx = 0.1
	sy = 0.05
	rect = patches.Rectangle((idx-sx/2,0-sy/2),sx,sy,linewidth=0.1,edgecolor='r',facecolor='none', color=color)
	ax.add_patch(rect)

sw_prob_logs = []
ss_logs = []
sub_sentences_logs = []
sub_sentences_range_logs = []
max_class_display = 20
def process_tw_prob(s, filename = 'result_output/tmp.png', draw = False, draw_sw_prob = False):

	processed_result = collections.defaultdict()

	ss = []
	ss.append('eos')
	ss.extend(s.split(' '))
	ss.append('eos')

	p_dists = transform(tw_model, ss)

	'''
	for p_dist_idx, p_dist in enumerate(p_dists):
		print(ss[p_dist_idx])
		for y_idx, p in enumerate(p_dist):
			if p > 0:
				print('    {} : {}'.format(str(y_labels[y_idx]), str(p)))
		print()
	'''

	max_p_dist = list(zip(y_labels, merge_tmp_us(p_dists)))
	max_p_dist = [ _ for _ in max_p_dist if _[1] > 0 ]
	#print(max_p_dist)
	max_p_dist = sorted(max_p_dist, key = lambda x : x[1])
	top_prob_tags = [ _[0] for _ in max_p_dist[-1*max_class_display:] ]

	top_prob_tags.append('sum')
	if draw_sw_prob:
		top_prob_tags.append('sw_prob')

	p_dists_T = transpose(p_dists)

	plt.clf()

	d_dict = collections.defaultdict()
	x = np.array([ _ for _ in range(len(ss))])
	d_dict['x'] = x
	d_dict['zero'] = np.array([ 0 for _ in range(len(ss))])
	plt.xticks(x, ss)

	sw_probs = [ [] for _ in range(len(ss)) ]
	sw_prob = zero_vec(len(ss))

	for _idx in range(len(ss)-1):
		conv_word = ' '.join([ss[_idx], ss[_idx+1]])
		if not stopword_probs.get(conv_word, None) is None:
			#draw_rect(plt, _idx, width = 2)
			sw_prob[_idx] += stopword_probs[conv_word]/2
			sw_prob[_idx+1] += stopword_probs[conv_word]/2

			sw_probs[_idx].append(stopword_probs[conv_word]/2)
			sw_probs[_idx+1].append(stopword_probs[conv_word]/2)


	for w_idx, w in enumerate(ss):
		if not stopword_probs.get(w, None) is None:
			#draw_rect(plt, w_idx)
			sw_prob[_idx] += stopword_probs[w]

	for _i in range(len(sw_prob)):
		if sw_prob[_i] > 0:
			sw_prob[_i] = math.log(sw_prob[_i])

	sw_breaks = np.array(sw_prob)
	break_threashold = 4
	sw_breaks[sw_breaks < break_threashold] = 0
	sw_breaks[sw_breaks >= break_threashold] = 1


	# Create a rangeset
	rs = rangeset.RangeSet(0, 0)

	for _i in range(len(sw_breaks)):
		if sw_breaks[_i]:
			rs = rs | rangeset.RangeSet(_i, _i+1)

	core_rs = rangeset.RangeSet(0, len(sw_breaks))
	new_rs = []
	for r in rs:
		r_len = (rangeset.RangeSet(r[0], r[1]).range())
		if r_len > 0:
			new_rs.append(r)
			draw_rect(plt, r[0], width = r[1] - r[0])
			core_rs = core_rs - r


	for y_idx, p_dist_T in enumerate(p_dists_T):
		y_label = y_labels[y_idx]
		y = np.array(p_dist_T)
		d_dict[y_label] = y

	sub_prob_serie_names = []
	sub_sentences = []
	sub_sentences_ranges = []
	sub_sentences_sub_classes = []
	for core_r in core_rs:
		s_idx = core_r[0] - 1
		e_idx = core_r[1] + 1
		s_idx = max(s_idx, 1)
		e_idx = min(e_idx, len(sw_breaks) - 1)
		sub_sent = ss[s_idx:e_idx]
		sub_sentences_ranges.append([s_idx - 1, e_idx - 1])
		if len(sub_sent) >= 2:
			sub_sentences.append(sub_sent)

			d_dict_slice_tmp = collections.defaultdict()
			d_dict_slice_tmp_sum = collections.defaultdict()
			d_dict_slice_tmp_max = collections.defaultdict()

			for y_idx, p_dist_T in enumerate(p_dists_T):
				y_label = y_labels[y_idx]
				d_dict_slice_tmp[y_label] = p_dist_T[s_idx:e_idx]
				d_dict_slice_tmp_sum[y_label] = sum(d_dict_slice_tmp[y_label])
				d_dict_slice_tmp_max[y_label] = max(d_dict_slice_tmp[y_label])

			d_dict_slice_tmp_num = d_dict_slice_tmp_sum
			d_dict_slice_tmp_num = list( sorted( list(d_dict_slice_tmp_num.items()), key = lambda x:x[1] ) )
			print(d_dict_slice_tmp_num[:5])
			print(d_dict_slice_tmp_num[-5:])
			print('-+'*10)
			print()

			sub_sentences_sub_classes.append(d_dict_slice_tmp_num[-1])

			max_prob = zero_vec(len(ss))
			subclass_tuple = d_dict_slice_tmp_num[-1]

			for _ in range(s_idx, e_idx):
				max_prob[_] = subclass_tuple[1]

			# add sum of prob
			sub_prob_serie_name = 'sum prob : '+subclass_tuple[0]
			d_dict[sub_prob_serie_name] = np.array(max_prob)
			sub_prob_serie_names.append(sub_prob_serie_name)

			# calculate most related topic.

		#print(sub_sent)

	processed_result['sub_sentences'] = sub_sentences
	processed_result['sub_sentences_ranges'] = sub_sentences_ranges
	processed_result['sub_sentences_sub_classes'] = sub_sentences_sub_classes

	sum_p_dist = [ sum([_[w_idx] for _ in p_dists_T[:]]) for w_idx in range(len(p_dists_T[0])) ]
	d_dict['sum'] = np.array(sum_p_dist)
	d_dict['sw_prob'] = np.array(sw_prob) / 3.0

	processed_result['sw_prob'] = sw_prob
	processed_result['ss'] = ss

	# Data
	df = pd.DataFrame(d_dict)
	
	# multiple line plot
	_cnt = 1
	if draw:
		for num, y_label in enumerate(list(reversed(top_prob_tags))[:12]):
			if y_label == 'sum':
				continue

			plt.plot('x', y_label, data=df, marker='', color=palette(_cnt), linewidth=2)
			#plt.fill_between(df['x'], df[y_label], df['zero'], where=df[y_label] > df['zero'], facecolor=palette(_cnt), interpolate=True)
			if sum(df[y_label]) == 0:
				print(ss)
				print(y_label)
				print('-'*10)
				continue
			centroid_idx = sum(df['x']*df[y_label])/sum(df[y_label])
			#draw_centroid(plt, centroid_idx, palette(_cnt))

			_cnt += 1
			
			sub_prob_serie_name = 'sum prob : '+y_label
			if sub_prob_serie_name in sub_prob_serie_names:
				plt.plot('x', sub_prob_serie_name, data=df, marker='', color=palette(_cnt), linewidth=2)

		# Get the current reference
		idx = 3
		plt.legend()
		plt.savefig(filename)
	return processed_result

import random

def get_random_s():
	try:
		r__y_idx = random.randint(0, len(y_labels) - 1)
		y_label = y_labels[ r__y_idx ]
		r__s_idx = random.randint(0, len(data_dict[y_label]) - 1)
		s = data_dict[y_label][ r__s_idx ]
		return s
	except Exception as e:
		return ''

def get_accuracy(es, sub_sentences_range):
	
	segment_points = []

	_idx_cntr = 0
	esws = []
	raw_ws = []
	for s in es:
		ws = s.split(' ')
		esws.append(ws)
		raw_ws.extend(ws)
		print(ws)
		# start index
		_idx_cntr
		segment_points.append(_idx_cntr)
		# end index
		_idx_cntr += len(ws)
		segment_points.append(_idx_cntr)

	segment_points = sorted(list(set(segment_points)))

	predict_segment_points = []
	for r in sub_sentences_range:
		predict_segment_points.append(r[0])
		predict_segment_points.append(r[1])

	predict_segment_points = sorted(list(set(predict_segment_points)))

	data_blocks = []
	data_block_mapping = collections.defaultdict()
	data_block_expected_result_mapping = collections.defaultdict()
	data_block_predicted_result_mapping = collections.defaultdict()

	for _ in range(len(raw_ws) + 1):
		data_block_expected_result_mapping[_] = False
		data_block_predicted_result_mapping[_] = False

		data_block_mapping[_] = len(data_blocks)
		data_blocks.append(_)
		if _ < len(raw_ws):
			data_blocks.append(raw_ws[_])
	

	print(list(zip(range(len(raw_ws)), raw_ws)))
	print(segment_points)
	print(predict_segment_points)

	for segment_point in segment_points:
		idx = data_block_mapping[segment_point]
		data_block_expected_result_mapping[segment_point] = True
		data_blocks[idx] = str(data_blocks[idx]) + ' | E'

	for predict_segment_point in predict_segment_points:
		idx = data_block_mapping[predict_segment_point]
		data_block_predicted_result_mapping[predict_segment_point] = True
		data_blocks[idx] = str(data_blocks[idx]) + ' | P'

	print('-'*10)
	for data_block in data_blocks:
		print(data_block)
	print('-'*10)

	print('Acc @ dist = 0')

	acc_result_dict = collections.defaultdict()
	acc_result_dict['TP'] = 0
	acc_result_dict['FP'] = 0
	acc_result_dict['FN'] = 0
	acc_result_dict['TN'] = 0

	for bp_idx in range(len(raw_ws) + 1):
		prev_bp_idx = max(bp_idx - 1, 0)
		next_bp_idx = min(bp_idx + 1, len(raw_ws))

		is_expt = data_block_expected_result_mapping[bp_idx]
		is_pred = data_block_predicted_result_mapping[bp_idx]

		tn_cond = not is_expt and not is_pred
		tp_cond = is_expt and is_pred
		fp_cond = not is_expt and is_pred
		fn_cond = is_expt and not is_pred

		acc_result = ''
		if tn_cond:
			acc_result = 'TN'
		if tp_cond:
			acc_result = 'TP'
		if fp_cond:
			acc_result = 'FP'
		if fn_cond:
			acc_result = 'FN'

		acc_result_dict[acc_result] = acc_result_dict.get(acc_result, 0)
		acc_result_dict[acc_result] += 1

	for key in sorted(acc_result_dict.keys()):
		print(key+" : "+str(acc_result_dict[key]))
	precision = acc_result_dict['TP'] / ( acc_result_dict['TP'] + acc_result_dict['FP'] )
	recall = acc_result_dict['TP'] / ( acc_result_dict['TP'] + acc_result_dict['FN'] )

	print('precision : '+str(precision))
	print('recall : '+str(recall))
	acc_result_dict['precision'] = precision
	acc_result_dict['recall'] = recall
	acc_dist_0 = acc_result_dict
	print('='*5)
	print('Acc @ dist = 1')

	# acc @ dist <= 1

	acc_result_dict = collections.defaultdict()
	acc_result_dict['TP'] = 0
	acc_result_dict['FP'] = 0
	acc_result_dict['FN'] = 0
	acc_result_dict['TN'] = 0

	for bp_idx in range(len(raw_ws) + 1):
		prev_bp_idx = max(bp_idx - 1, 0)
		next_bp_idx = min(bp_idx + 1, len(raw_ws))

		is_expt = data_block_expected_result_mapping[bp_idx]
		is_pred = data_block_predicted_result_mapping[bp_idx]

		prev_is_expt = data_block_expected_result_mapping[prev_bp_idx]
		prev_is_pred = data_block_predicted_result_mapping[prev_bp_idx]

		next_is_expt = data_block_expected_result_mapping[next_bp_idx]
		next_is_pred = data_block_predicted_result_mapping[next_bp_idx]

		tn_cond = ( not is_expt or ( is_expt and prev_is_pred ) or ( is_expt and next_is_pred ) ) and not is_pred
		tp_cond = ( is_expt or prev_is_expt or next_is_expt ) and is_pred
		fp_cond = ( not is_expt and not prev_is_expt and not next_is_expt ) and is_pred
		fn_cond = ( is_expt and not prev_is_pred and not next_is_pred ) and not is_pred

		acc_result = ''
		if tn_cond:
			acc_result = 'TN'
		if tp_cond:
			acc_result = 'TP'
		if fp_cond:
			acc_result = 'FP'
		if fn_cond:
			acc_result = 'FN'

		acc_result_dict[acc_result] = acc_result_dict.get(acc_result, 0)
		acc_result_dict[acc_result] += 1

	for key in sorted(acc_result_dict.keys()):
		print(key+" : "+str(acc_result_dict[key]))

	precision = acc_result_dict['TP'] / ( acc_result_dict['TP'] + acc_result_dict['FP'] )
	recall = acc_result_dict['TP'] / ( acc_result_dict['TP'] + acc_result_dict['FN'] )
	print('precision : '+str(precision))
	print('recall : '+str(recall))
	acc_result_dict['precision'] = precision
	acc_result_dict['recall'] = recall
	acc_dist_1 = acc_result_dict
	print('='*5)
	return { 'acc_dist_0' : acc_dist_0 , 'acc_dist_1' : acc_dist_1 }

print('='*10)

import sh
files = glob.glob('result_output/*')
print(files)
#sh.rm(files)
for f in files:
	print(f)
	os.remove(f)

print('='*10)

q_results = []
for __ in range(1, 4):
	for _ in range(20):

		es = []
		while len(es) <  __:
			rand_s = get_random_s()
			if len(rand_s.strip()) > 0:
				es.append(rand_s)

		s = ' '.join(es)
		process_tw_prob(s, filename = 'result_output/wosw_tmp_{}s_{}.png'.format(str(__), _), draw = True, draw_sw_prob = False)
		processed_result = process_tw_prob(s, filename = 'result_output/wsw_tmp_{}s_{}.png'.format(str(__),_), draw = True, draw_sw_prob = True)
		print(es)
		sub_sentences_ranges = processed_result['sub_sentences_ranges']
		acc_dict = get_accuracy(es, sub_sentences_ranges)
		print(acc_dict)

		acc_dist_0 = acc_dict['acc_dist_0']
		acc_dist_1 = acc_dict['acc_dist_1']

		sub_sentences = [ ' '.join(_) for _ in processed_result['sub_sentences'] ]
		sub_sentences_sub_classes = [ str(_) for _ in processed_result['sub_sentences_sub_classes'] ]
		merged_sub_sentences = zip(sub_sentences, sub_sentences_sub_classes)
		merged_sub_sentences = [ _[0]+','+_[1] for _ in merged_sub_sentences ]

		tmp = collections.defaultdict()
		tmp['expected_segment'] = '|'.join(es)
		tmp['predict_segment'] = '|'.join([ _ for _  in merged_sub_sentences ])

		tmp['input_sentence'] = s
		tmp['precision_dist_0'] = acc_dist_0['precision']
		tmp['recall_dist_0'] = acc_dist_0['recall']
		tmp['precision_dist_1'] = acc_dist_1['precision']
		tmp['recall_dist_1'] = acc_dist_1['recall']
		q_results.append(tmp)

import csv

with open('result_output/tw_segment_result.csv', 'w') as fo:
	headers = ['expected_segment', 'predict_segment', 'input_sentence', 'precision_dist_0', 'recall_dist_0', 'precision_dist_1', 'recall_dist_1']
	writer = csv.DictWriter(fo, fieldnames = headers)
	writer.writeheader() 
	for q_result in q_results:
		writer.writerow(q_result)
		fo.flush()
	fo.close()











