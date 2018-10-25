''' 
Tests for inferrer
'''
import numpy as np
from clmm.inferrer import parameter_sampler as ps
from clmm.inferrer import collection as cll
from numpy import testing as tst

gc_objects_input = {i:{'rich':10+i/10} for i in range(100)}
bins_specs = [{'rich':(i, i+10)} for i in range(10, 101, 10)]

def test_ps_init():
    ps.ParameterSampling(None, None)
    pass

def test_ps_run():
    psamp = ps.ParameterSampling(None, None)
    psamp.run(None)
    pass

def test_cll__name_bin():
    collection = cll.Collections(bin_specs)
    for bin_spec in bins_specs:
        expected_name = 'rich'+str(bin_spec['rich'])
        tst.assert_equal(collection._name_bin(bin_spec), expected_name)

def test_cll_get_cl_in_bins():
    collection = cll.Collections(bin_specs)
    bin_collction = collection.get_cl_in_bins(gc_objects_input)
    for gc, gc_data in gc_objects_input.items():
        floor = np.floor(gc_data['rich']/10.)*10
        expected_name = 'rich(%d, %d)'%(floor, floor+10)

        assert expected_name in bin_collection
        assert gc in bin_collection[expected_name]
