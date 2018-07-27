''' This contains the galaxy cluster class, each instance corresponds to a galaxy cluster in our dataset.'''

class GalaxyCluster(object) :
    '''Galaxy cluster: reads in data, datatype... '''
    def __init__(self, data, datatype) :
        self.data = data
        self.datatype = datatype
        
