def zero_vec(length):
	return [ 0 for _ in range(length) ]

# max of probability classes (y_labels)
def merge_vec(u, v):
	if not len(u) == len(v):
		return None
	w = zero_vec(len(u))
	for i in range(len(u)):
		w[i] = max(u[i], v[i])
	return w

# max of probability classes (y_labels)
def sum_vec(u, v):
	if not len(u) == len(v):
		return None
	w = zero_vec(len(u))
	for i in range(len(u)):
		w[i] = u[i] + v[i]
	return w

def transpose(mat):
	return [*zip(*mat)]