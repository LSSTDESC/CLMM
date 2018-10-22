''' 
Tests for manager
'''
import os, sys
DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/'.join(DIR.split('/')[:-1]))

from clmm.core import manager
#from test_galaxycluster import *
from clmm.core.datatypes import *

test_dict = {'test%d'%i:True for i in range(3)}
test_table = []
test_data = GCData('test_data', test_dict, test_table)
manager_guy = mangager.Manager()

def func_test(data):
    print '*** Here is your data ***'
    print data
    print '*************************'
    return

from numpy import testing as tst

def test_signcreator():
    manager_guy._signcreator(func_test)
    pass

def test_signspecs():
    pass

def test_apply() :
    pass

def test_prepare() :
    pass

def test_deliver() :
    pass

