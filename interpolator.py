import math
import numpy as np

from typing import List

from fancyimpute import KNN as knn
from fancyimpute import NuclearNormMinimization as nnm


class FieldCaster:
    
    m_mtx_probe = List[List[float]]
    m_mtx_inter = List[List[float]]
    
    def __init__(self, mtx : List[List[float]]):
        
        self.m_mtx_probe = mtx
        self.m_mtx_inter = nnm().fit_transform(self.m_mtx_probe)
        
        
    def get_interpolated(self) -> List[List[float]]:
        return self.m_mtx_inter