from math import *
import numpy as np

#import matplotlib.pyplot as plt



w_true = [ 1.0, 2.0, 3.0, 4.0, 5.0 ]
input_params = [ [0.0, 0.1], [0.0, 0.1], [0.0, 0.1], [0.0, 0.1] ]
input_dist = [ "normal", "normal", "normal", "normal" ]
noise_params = [ 0.0, 0.1 ]
noise_dist = "normal" 

N_inputs = 4
N_samples = 100
R_mat = np.identity(N_samples)
N_iter = 100

global Debug_tmp
Debug_tmp = None
Doc = lambda obj: print(obj.__doc__)
IsList = lambda obj: "list" in str(type(obj))


def f(w, u):
	return w[0] + w[1]*u[0] + w[2]*u[1] + w[3]*u[2] + w[4]*u[3]

f_t = lambda u: f(w_true, u)

def show_obj():
	tmp = "".join(" %f*u" + str(i) + " +" for i in range(1, len(w_true)))
	print(("y = %f +" + tmp[:-1]) % tuple(w_true))

def show_true_w():
	print("True params is:")
	print(w_true)

def show_inputs_info():
	print("_"*80)
	print("INPUTS INFO:")
	print("Num of inputs: ", N_inputs)
	print("Num of samples: ", N_samples)
	print("Max iterations: ", N_iter)
	for i in range(0, N_inputs):
		Str_f = ""
		Str_f += input_dist[i][0].upper()
		Str_f += str(tuple(input_params[i]))		
		print("%d: U%d = %s" % (i, i, Str_f))
	print("\nNOISE INFO:")
	print("Noise dist: ", noise_dist)
	print("Noise dist params: ", noise_params)
	print("_"*80)

def gen_u_(sz, params = [ 0.0, 1.0 ], dist = "normal"):
	if params == None:
		params = [ 0.0, 1.0 ]
	if dist == None:
		dist = "normal"
	cmd = "np.random." + dist + "(" + str(params[0]) + ", " +\
		str(params[1]) + ", " + str(sz) + ")"
	u_ = eval(cmd)
	return u_

def gen_u(sz, params_a = None, dist_a = None):
	N_samples = sz[0]
	N_inputs = sz[1]
	if params_a == None:
		params_a = N_inputs*[None]
	if dist_a == None:
		dist_a = N_inputs*[None]
	U = []
	tmp = []
	for i in range(0, N_inputs):
		tmp = gen_u_(N_samples, params_a[i], dist_a[i])
		U.append(tmp)
	U = np.array(U)
	return U.T

def gen_n_(sz = 1, params = [0.0, 1.0], dist = "normal"):
	if params == None:
		params = [ 0.0, 1.0 ]
	if dist == None:
		dist = "normal"
	cmd = "np.random." + dist + "(" + str(params[0]) + ", " +\
		str(params[1]) + ", " + str(sz) + ")"
	n_ = eval(cmd)
	return n_

def gen_n(sz, params, dist):
	N = []
	for i in range(0, N_samples):
			N.append(gen_n_(1, params, dist)[0])
	N = np.array(N)
	return N

def calc_ft(w = w_true, U = [], func = f_t):
	if U == []:
		return []
	N_samples = len(U)
	Y = []
	for u in U:
		Y.append(func(u))
	Y = np.array(Y)
	return Y

def calc_f(w, U = [], func = f):
	if U == []:
		return []
	N_samples = len(U)
	Y = []
	for u in U:
		Y.append(func(w, u))
	Y = np.array(Y)
	return Y

def calc_J(w, U):
	tmp = [ R_mat[i][i]*(f(w, u) - f_t(u))**2 for i, u in enumerate(U) ]
	return np.sum(tmp)


def calc_w(U, Y, R):
	tmp = np.dot(U.T, R_mat)
	tmp2 = np.dot(tmp, Y)
	tmp = np.dot(tmp, U)
	tmp = np.dot(np.linalg.inv(tmp), tmp2)
	return tmp

U_values = gen_u((N_samples, N_inputs), input_params, input_dist)
N_values = gen_n(N_samples, noise_params, noise_dist)
Y_true = calc_ft(U = U_values)
Y_true_n = Y_true + N_values

U_values_m = np.sum(U_values, axis = 0)/len(U_values)
Y_true_n_m = np.sum(Y_true_n)/len(Y_true_n)

U_values_0 = np.array([ (u - U_values_m).tolist() for u in U_values ])
Y_true_n_0 = Y_true_n - Y_true_n_m

new_w = calc_w(U_values_0, Y_true_n_0, R_mat)
tmp_w = [ 0.0 ] + new_w.tolist()
new_Y_0 = f([ 0.0 ] + new_w.tolist(), U_values_m)
new_w_0 = Y_true_n_m - new_Y_0
new_w = [ new_w_0 ] + new_w.tolist()
w_errors = np.abs(np.array(new_w) - np.array(w_true))
show_true_w()
show_obj()
show_inputs_info()

print("Params estimates by mls: ")
print(new_w)
print("_"*80)
print("J value: ", calc_J(tmp_w, U_values_0))
print("Abs errors: ", w_errors.tolist())
print("Abs sum squared error: ", np.sum([ err**2 for err in w_errors ])/len(w_errors))

