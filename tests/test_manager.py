''' 
Tests for manager
'''

from clmm.core import manager
from clmm.core.datatypes import *
from clmm.galaxycluster import *

test_spec = {'test%d'%i:True for i in range(3)}
test_table = []
test_data = GCData('test_data', test_spec, test_table)
test_data_out = GCData('func_test', test_spec, test_table)

manager_guy = manager.Manager()

test_gc = GalaxyCluster(test_data)
test_gc_out = GalaxyCluster(test_data)
test_gc_out.add_data(test_data_out)

def func_test(data, **argv):
    print('*** Here is your data ***')
    print(data)
    print('* and aux args:')
    for a, i in argv.items():
        print(a, i)
    print('*************************')
    return

from numpy import testing as tst

def test_signcreator():
    tst.assert_equal(manager_guy._signcreator(func_test), 'func_test')

def test_signspecs():
    tst.assert_equal(manager_guy._signspecs(test_spec), test_spec)

def test_pack() :
    tst.assert_equal(manager_guy._pack(func_test, test_spec, []), test_data_out)

def test_unpack() :
    tst.assert_equal(manager_guy._unpack(test_gc, func_test, test_spec), test_data)

def test_apply() :
    manager_guy.apply(test_gc, func_test, test_spec)
    #tst.assert_equal(test_gc, test_gc_out)
    for d1, d2 in zip(test_gc.data, test_gc_out.data):
        print(d1, d2)
        tst.assert_equal(d1, d2)

def test_prepare() :
    pass

def test_deliver() :
    pass

