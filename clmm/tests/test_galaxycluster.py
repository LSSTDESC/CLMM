import os
print(os.path.dirname(os.path.realpath(__file__)))

import datatypes
#import clusterdatatypes
#from galaxycluster import *
#import datatypes
from datatypes import GCData, find_in_dataset

def print_title(title, space=True):

    line = '#'.join(['' for i in range(15+len(title))])

    print(line)
    print('###### %s ######'%title)
    print(line)
    print()

def print_job(job):

    print('>>>', job)
    print()

test_dict = {'test%d'%i:True for i in range(3)}
test_dict_fal = {'test%d'%i:False for i in range(3)}
test_dict_sub = {'test%d'%i:True for i in range(2)}
test_data = GCData('Mitch', test_dict,[1, 2, 3])



import numpy as np

def test_check_subdict():

    np.testing.asserts_equal(True, check_subdict(test_dict_sub, test_dict))
    np.testing.asserts_equal(False, check_subdict(test_dict_sub, test_dict_fal))
    
'''
#################################################
################ test Aux #######################
#################################################
print_title('--- test Aux ---')

######## _check_subdict
print()
print_title('_check_subdict')

test_gen(True, check_subdict, test_dict_sub, test_dict)
test_gen(False, check_subdict, test_dict_sub, test_dict_fal)

######## _find_in_datalist
print()
print_title('_find_in_datalist')

print_job( find_in_datalist(test_dict    , gc.data['Mitch'], )  )
print_job( find_in_datalist(test_dict_sub, gc.data['Mitch'], )  )
print_job( find_in_datalist(test_dict_fal, gc.data['Mitch'], )  )

######## _find_ind_in_datalist_exact
print()
print_title('_find_ind_in_datalist_exact')

print_job( find_ind_in_datalist_exact( gc.data['Mitch'], test_dict)  )
print_job( find_ind_in_datalist_exact( gc.data['Mitch'], test_dict_sub)  )
print_job( find_ind_in_datalist_exact( gc.data['Mitch'], test_dict_fal)  )

######## _find_in_datalist_exact
print()
print_title('_find_in_datalist_exact')

print_job( find_in_datalist_exact( gc.data['Mitch'], test_dict)  )
print_job( find_in_datalist_exact( gc.data['Mitch'], test_dict_sub)  )
print_job( find_in_datalist_exact( gc.data['Mitch'], test_dict_fal)  )

######## _remove_in_datalist
print()
print_title('_remove_in_datalist')

print_job( remove_in_datalist( gc.data['Mitch'], test_dict)  )
print_job( gc.data['Mitch'])
print_job( gc._add( test_data)  )
print_job( gc.data['Mitch'])
print_job( remove_in_datalist( gc.data['Mitch'], test_dict_fal)  )
print_job( remove_in_datalist( gc.data['Mitch'], test_dict_sub)  )

###########################################################
################ test GalaxyCluster #######################
###########################################################
print_title('--- test GalaxyCluster ---')

######## _check_datatype
print()
print_title('_check_datatype')

print_job( gc._check_datatype('Mitch')  )
print_job( gc._check_datatype(test_data)  )

######## _check_creator
print()
print_title('_check_creator')

print_job( gc._check_creator('Mitch')  )
print_job( gc._check_creator('Joe')  )

######## _find
print()
print_title('_find')

print_job( gc._find( 'Joe', test_dict)  )
print_job( gc._find( 'Mitch', test_dict)  )
print_job( gc._find( 'Mitch', test_dict_sub)  )
print_job( gc._find( 'Mitch', test_dict_fal)  )

######## _find_exact
print()
print_title('_find_exact')

print_job( gc._find_exact( 'Joe', test_dict)  )
print_job( gc._find_exact( 'Mitch', test_dict)  )
print_job( gc._find_exact( 'Mitch', test_dict_sub)  )
print_job( gc._find_exact( 'Mitch', test_dict_fal)  )

######## _add
print()
print_title('_add')

print_job( gc._add( test_data)  )
print_job( gc._add( test_data, True)  )

######## _remove
print()
print_title('_remove')

print_job( gc._remove( 'Joe', test_dict)  )
print_job( gc._remove( 'Mitch', test_dict)  )
print_job( gc.data['Mitch'])
print_job( gc._add( test_data)  )
print_job( gc.data['Mitch'])
print_job( gc._remove( 'Mitch', test_dict_fal)  )
print_job( gc._remove( 'Mitch', test_dict_sub)  )
'''
