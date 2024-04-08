import math
import numpy as np

from typing import List

from fancyimpute import KNN as knn
from fancyimpute import NuclearNormMinimization as nnm


class Interpolator:
    
    m_mtx_probe = List[List[int]]
    m_mtx_inter = List[List[int]]
    
    def __init__(self, mtx : List[List[int]]):
        
        self.m_mtx_probe = mtx
        self.m_mtx_inter = nnm().fit_transform(self.m_mtx_probe)
        
        
    def get_interpolated(self) -> List[List[int]]:
        return self.m_mtx_inter