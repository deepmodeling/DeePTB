from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass


class AbstractStructure(with_metaclass(ABCMeta, object)):
    '''This class is an abstraction class for all instances
    that requires a atomic structure, such as TBHamiltonian'''

    def __init__(self):
        self.bondlist = None
        self.struct = None

    def _set_bond(self, bondlist):
        self.bondlist = bondlist

    def _set_struct(self, struct):
        self.struct = struct

    @abstractmethod
    def cal_bond(self):
        '''
        get the bondList from loaded struct

        '''

        pass

    @abstractmethod
    def read_struct(self, struct, type):
        '''
        read struct using ASE
        '''
        pass