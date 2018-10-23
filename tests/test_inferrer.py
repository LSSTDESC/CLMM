''' 
Tests for inferrer
'''
import numpy as np
from clmm.inferrer import inferrer
from numpy import testing as tst

gc_objects_input = {i:{'rich':10+i/10} for i in range(100)}
bins_specs = [{'rich':(i, i+10)} for i in range(10, 101, 10)]

def test_init():
    inferrer_guy = inferrer.Inferrer(gc_objects_input)
    for gc, gc_data in gc_objects_input.items():
        tst.assert_equal(inferrer_guy.gc_objects[gc]['data'], gc_data)

def test__name_bin():
    inferrer_guy = inferrer.Inferrer(gc_objects_input)
    for bin_spec in bins_specs:
        expected_name = 'rich'+str(bin_spec['rich'])
        tst.assert_equal(inferrer_guy._name_bin(bin_spec), expected_name)

def test__add_bin_info():
    '''
    Adds information about which bin each cluster belongs to gc_objects

    Parameters
    ----------
    bins_specs: list
        List with specifications for each bin
    '''
    '''
    for name, obj in self.gc_objects.items():
        for bin_spec in bins_specs:
            if obj['data'] in bin:# to complete this
                self.gc_objects[name]['bin'] = \
                        self._name_bin(bin_spec)
                break
    '''
    inferrer_guy = inferrer.Inferrer(gc_objects_input)
    inferrer_guy._add_bin_info(bins_specs)
    for gc, gc_data in gc_objects_input.items():
        floor = np.floor(gc_data['rich']/10.)*10
        expected_name = 'rich(%d, %d)'%(floor, floor+10)
        tst.assert_equal(inferrer_guy.gc_objects[gc]['bin'], expected_name)

def test_run_gc_chains():
    '''
    Runs chains for constraining parameters of each cluster
    individially
    '''
    #self.gc_chains[gc.name] = some_function(self.gc_objects)
    pass

def test_run_is_chains():
    '''
    Runs Important Sampling on all given bins. It is required that 
    all self.gc_objects in all bins be filled with individual chains.

    Parameters
    ----------
    bins_specs: list
        List with specifications for each bin
    '''
    '''
    self._add_bin_info(bins_specs)
    for bin_spec in bins_specs:
        self._run_is_chain(bin_spec)
    '''
    pass

def test__run_is_chain():
    '''
    Runs Important Sampling on one specific bin. It is required that 
    all self.gc_objects in this bin be filled with individual chains.

    Parameters
    ----------
    bin_spec: ???
        Specifications for a certain bin
    '''
    '''
    bin_name = self._name_bin(bin_spec)
    collection_chains = [obj['chain'] for obj in self.gc_objects.values() \
                                if obj['bin'] == bin_name]
    #self.is_chains[bin_name] = some_function(collection_chains)
    '''
    pass
