
run_my_solution = False

import os
import copy
import signal
import numpy as np
import shlex

if run_my_solution:
    # from A3mysolution import *
    import neuralnetworkA3 as nn
    # from neuralnetworkA3 import NeuralNetwork
    print('##############################################')
    print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    print('##############################################')

else:
    
    import subprocess, glob, pathlib, platform

    assignmentNumber = '3'

    nb_name = f'A{assignmentNumber}[Ss]olution*.ipynb'
    # nb_name = '*.ipynb'
    filename = next(glob.iglob(nb_name), None)

    print('\n======================= Code Execution =======================\n')

    print('Extracting python code from notebook named \'{}\' and storing in notebookcode.py'.format(filename))
    if not filename:
        raise Exception(f'Please rename your notebook file to A{assignmentNumber}solution.ipynb'.format(assignmentNumber))

    with open('notebookcode.py', 'w') as outputFile:
        on_windows = platform.system() == "Windows"
        cmd = "where" if on_windows else "which"
        res = subprocess.run([cmd, 'jupyter'], capture_output=True)
        jup = res.stdout[:-1].decode('utf-8')
        comm = f'{jup} nbconvert --to script {nb_name} --stdout --Application.log_level=WARN'
        # print(shlex.split(comm))
        if on_windows:
            subprocess.call(shlex.split(comm), stdout=outputFile, shell=True)
        else:
            subprocess.call(shlex.split(comm), stdout=outputFile)

    import sys
    import ast
    import types
    with open('notebookcode.py') as fp:
        tree = ast.parse(fp.read(), 'eval')
    print('Removing all statements that are not function or class defs or import statements.')
    for node in tree.body[:]:
        if (not isinstance(node, ast.FunctionDef) and
            not isinstance(node, ast.Import) and
            not isinstance(node, ast.ClassDef)):
            # not isinstance(node, ast.ImportFrom)):
            tree.body.remove(node)
    # Now write remaining code to py file and import it
    module = types.ModuleType('notebookcodeStripped')
    code = compile(tree, 'notebookcodeStripped.py', 'exec')
    sys.modules['notebookcodeStripped'] = module
    exec(code, module.__dict__)
    # import notebookcodeStripped as useThisCode
    from notebookcodeStripped import *


def test(points, runthis, correct_str, incorrect_str):
    if (runthis):
        print()
        print('-'*70)
        print(f'----  {points}/{points} points. {correct_str}')
        print('-'*70)
        return points
    else:
        print()
        print('-'*70)
        print(f'----  0/{points} points. {incorrect_str}')
        print('-'*70)
        return 0

for func in ['NeuralNetwork']:
    if func not in dir(nn):  #  or not callable(globals()[func]):
        print('CRITICAL ERROR: Class named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')
        break
    for method in ['_forward', 'get_error_trace', 'gradient_f',
                    'error_f', '_make_weights_and_views', 'train', 'use']:
        if method not in dir(nn.NeuralNetwork):
            print('CRITICAL ERROR: NeuralNetwork Function named \'{}\' is not defined'.format(method))
            print('  Check the spelling and capitalization of the function name.')
            


import neuralnetworkA3 as nn
    
exec_grade = 0

######################################################################

runthis = '''
def check_weight_views(nnet):
    results = []
    for layeri, W in enumerate(nnet.Ws):
        if np.shares_memory(nnet.all_weights, W):
            print(f'nnet.Ws[{layeri}] correctly shares memory with nnet.all_weights')
            results.append(True)
        else:
            print(f'nnet.Ws[{layeri}] does not correctly share memory with nnet.all_weights')
            results.append(False)

    return np.all(results)

n_inputs = 3
n_hiddens = [12, 8, 4]
n_outputs = 2

nnet = nn.NeuralNetwork(n_inputs, n_hiddens, n_outputs)
'''

testthis = 'check_weight_views(nnet)'

pts = 5

try:
    print('\n')
    print('='*80, f'\nTesting this for {pts} points:')
    print(runthis)
    exec(runthis)
    print(f'\n#  and test result with    {testthis}')
    exec_grade += test(pts, testthis,
                       'Weight views are correctly defined',
                       'Weight views are not correctly defined.')
except Exception as ex:
    print(f'\n--- 0/{pts} points. NeuralNetwork constructor raised the exception\n')
    print(ex)


######################################################################

runthis = '''
nnet = nn.NeuralNetwork(3, [], 4)
'''

testthis = 'check_weight_views(nnet)'

pts = 5

try:
    print('\n')
    print('='*80, f'\nTesting this for {pts} points:')
    print(runthis)
    exec(runthis)
    print(f'\n#  and test result with    {testthis}')
    exec_grade += test(pts, testthis,
                       'Weight views are correctly defined',
                       'Weight views are not correctly defined.')
except Exception as ex:
    print(f'\n--- 0/{pts} points. NeuralNetwork constructor raised the exception\n')
    print(ex)

######################################################################

runthis = '''
def check_gradient_views(nnet):
    results = []
    for layeri, G in enumerate(nnet.Grads):
        if np.shares_memory(nnet.all_gradients, G):
            print(f'nnet.Grads[{layeri}] correctly shares memory with nnet.all_gradients')
            results.append(True)
        else:
            print(f'nnet.Grads[{layeri}] does not correctly share memory with nnet.all_gradients')
            results.append(False)

    return np.all(results)

n_inputs = 3
n_hiddens = [5, 10, 20]
n_outputs = 2

nnet = nn.NeuralNetwork(n_inputs, n_hiddens, n_outputs)
'''

testthis = 'check_gradient_views(nnet)'

pts = 5


try:
    print('\n')
    print('='*80, f'\nTesting this for {pts} points:')
    print(runthis)
    exec(runthis)
    print(f'\n#  and test result with    {testthis}')
    exec_grade += test(pts, testthis,
                       'Gradient views are correctly defined',
                       'Gradient views are not correctly defined.')
except Exception as ex:
    print(f'\n--- 0/{pts} points. NeuralNetwork constructor raised the exception\n')
    print(ex)



######################################################################

runthis = '''
n_inputs = 3
n_hiddens = [5, 10, 20]
n_outputs = 2
n_samples = 10

X = np.arange(n_samples * n_inputs).reshape(n_samples, n_inputs) * 0.1
    
nnet = nn.NeuralNetwork(n_inputs, n_hiddens, n_outputs)
nnet.all_weights[:] = 0.1  # set all weights to 0.1
nnet.X_means = np.mean(X, axis=0)
nnet.X_stds = np.std(X, axis=0)
nnet.T_means = np.zeros((n_samples, n_outputs))
nnet.T_stds = np.ones((n_samples, n_outputs))
    
Y = nnet.use(X)

Y_answer = np.array([[0.14629519, 0.14629519],
                     [0.24029528, 0.24029528],
                     [0.33910878, 0.33910878],
                     [0.43981761, 0.43981761],
                     [0.53920896, 0.53920896],
                     [0.63421852, 0.63421852],
                     [0.72233693, 0.72233693],
                     [0.80186297, 0.80186297],
                     [0.87195874, 0.87195874],
                     [0.93254   , 0.93254   ]])
'''

testthis = 'np.allclose(Y, Y_answer, 0.1)'

pts = 15

try:
    print('\n')
    print('='*80, f'\nTesting this for {pts} points:')
    print(runthis)
    exec(runthis)
    print(f'\n#  and test result with    {testthis}')
    exec_grade += test(pts, testthis,
                       'nnet.use returned correct values.',
                       'nnet.use returned incorrect values.')

except Exception as ex:
    print(f'\n--- 0/{pts} points. NeuralNetwork constructor or use raised the exception\n')
    print(ex)


######################################################################

runthis = '''
n_inputs = 3
n_hiddens = [6, 3]
n_samples = 5

X = np.arange(n_samples * n_inputs).reshape(n_samples, n_inputs) * 0.1
T = np.log(X + 0.1)
n_outputs = T.shape[1]

def rmse(A, B):
    return np.sqrt(np.mean((A - B)**2))

results = []
for rep in range(20):
    nnet = nn.NeuralNetwork(n_inputs, n_hiddens, n_outputs)
    nnet.train(X, T, X, T, 2000, batch_size=-1, method='adamw', learning_rate=0.001, verbose=False)
    Y = nnet.use(X)
    err = rmse(Y, T)
    print(f'Net {rep+1} RMSE {err:.5f}')
    results.append(err)

mean_rmse = np.mean(results)
print(mean_rmse)
'''

testthis = '0.0 < mean_rmse < 0.1'

pts = 20

try:
    print('\n')
    print('='*80, f'\nTesting this for {pts} points:')
    print(runthis)
    exec(runthis)
    print(f'\n#  and test result with    {testthis}')
    exec_grade += test(pts, testthis,
                       'mean_rmse is correct value.',
                       'mean_rmse is incorrect value.')
except Exception as ex:
    print(f'\n--- 0/{pts} points. NeuralNetwork constructor, train, or use raised the exception\n')
    print(ex)

######################################################################

runthis = '''
n_inputs = 3
n_hiddens = [10, 10, 5]
n_samples = 5

X = np.arange(n_samples * n_inputs).reshape(n_samples, n_inputs) * 0.1
T = 2 + np.log(X + 0.1)
Xval = X + np.random.normal(0.0, 0.1, size=X.shape)
Tval = 2.1 + np.log(Xval + 0.1)
n_outputs = T.shape[1]
    
def rmse(A, B):
    return np.sqrt(np.mean((A - B)**2))

results = []
for rep in range(20):
    nnet = nn.NeuralNetwork(n_inputs, n_hiddens, n_outputs)
    nnet.train(X, T, Xval, Tval, 3000, batch_size=-1, method='adamw', learning_rate=0.1, verbose=False)
    Y = nnet.use(X)
    err = rmse(Y, T)
    print(f'Net {rep+1} RMSE {err:.5f} best epoch {nnet.best_epoch}')
    results.append(err)

mean_rmse = np.mean(results)
print(mean_rmse)
'''

testthis = '0.005 < mean_rmse < 0.2'

pts = 20

try:
    print('\n')
    print('='*80, f'\nTesting this for {pts} points:')
    print(runthis)
    exec(runthis)
    print(f'\n#  and test result with    {testthis}')
    exec_grade += test(pts, testthis,
                       'mean_rmse returned correct value.',
                       'mean_rmse returned incorrect value.')

except Exception as ex:
    print(f'\n--- 0/{pts} points. NeuralNetwork constructor, train, or use raised the exception\n')
    print(ex)


    
name = os.getcwd().split('/')[-1]

print()
print('='*70)
print(f'{name} Execution Grade is {exec_grade} / 70')
print('\n REMEMBER, YOUR FINAL EXECUTION GRADE MAY BE DIFFERENT,\n BECAUSE DIFFERENT TESTS WILL BE RUN.')
print('='*70)


print('''
Application Results:

___ / 10  1. Train with each of the three optimization, plot the error_traces for each
             of the three methods.  Discussion of what you see in the plots.
___ / 10  2. Use nested for loops to test various parameter values. Collect results in a
             DataFrame. Discussion of the set of parameter values and all three RMSE values
             that produce some of the lowest test RMSEs
___ / 10  3. Using best parameter values found, plot predicted critical temperature versus 
             the actual (target) critical temperatures for the training, validation, and test sets.
             Discuss what you see. How well does your neural network predict the critical temperatures?''')

print()
print('='*70)
print('{} Experiments and Discussion Grade is __ / 30'.format(name))
print('='*70)



print()
print('='*70)
print('{} FINAL GRADE is  _  / 100'.format(name))
print('='*70)

print('''
Extra Credit:  Code and discussion showing most significant input features, and results after removing half of the least significant features.
''')

print('\n{} EXTRA CREDIT is 0 / 1'.format(name))

if run_my_solution:
    print('##############################################')
    print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    print('##############################################')
    pass

