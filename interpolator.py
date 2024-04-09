import math
import numpy as np

from enum import Enum
from typing import List

from fancyimpute import KNN as knn
from fancyimpute import NuclearNormMinimization as nnm


class InterpolationType(Enum):
    
    K_NEAREST = 1
    NUC_NORM_MIN = 2
    SOFT_THOLD_SVD = 3
    ITERATIVE_ROUND = 4
    ITERATIVE_LOW_RANK_SVD = 5
    MATRIX_FACTORIZATION = 6
    BI_SCALER = 6
    ADAPTIVE_METHOD = 7
    

class Interpolator:
    
    m_mtx_probe = List[List[int]]
    m_mtx_inter = List[List[int]]
    
    def __init__(self, mtx : List[List[int]], type = InterpolationType.NUC_NORM_MIN):
        
        self.m_mtx_probe = mtx
        
        match type:
            case InterpolationType.K_NEAREST:
                self.m_mtx_inter = Interpolator.k_nearest_neighbors(mtx)
            case InterpolationType.NUC_NORM_MIN:
                self.m_mtx_inter = Interpolator.nuclear_norm_minimization(mtx)
            case InterpolationType.SOFT_THOLD_SVD:
                self.m_mtx_inter = Interpolator.soft_thold_single_val_decomp(mtx)
            case InterpolationType.ITERATIVE_ROUND:
                self.m_mtx_inter = Interpolator.iterative_round_impute(mtx)
            case InterpolationType.ITERATIVE_LOW_RANK_SVD:
                self.m_mtx_inter = Interpolator.iterative_single_val_decomp(mtx)
            case InterpolationType.MATRIX_FACTORIZATION:
                self.m_mtx_inter = Interpolator.factorization(mtx)
            case InterpolationType.BI_SCALER:
                self.m_mtx_inter = Interpolator.bi_scaler(mtx)
            case InterpolationType.ADAPTIVE_METHOD:
                self.m_mtx_inter = Interpolator.adaptive_method(mtx)
        
        
    def get_interpolated(self) -> List[List[int]]:
        return self.m_mtx_inter
    
    
    @staticmethod
    def k_nearest_neighbors(mtx : List[List[int]]) -> List[List[int]]:
        
        return mtx
    
    
    @staticmethod
    def nuclear_norm_minimization(probe : List[List[int]]) -> List[List[int]]:
        mtx = nnm().fit_transform(probe)
        return mtx
    
    
    @staticmethod
    def soft_thold_single_val_decomp(mtx : List[List[int]]) -> List[List[int]]:
        
        return mtx
    
    
    @staticmethod
    def iterative_round_impute(mtx : List[List[int]]) -> List[List[int]]:
        
        return mtx
    
    
    @staticmethod
    def iterative_single_val_decomp(mtx : List[List[int]]) -> List[List[int]]:
        
        return mtx
    
    
    @staticmethod
    def factorization(mtx : List[List[int]]) -> List[List[int]]:
        
        return mtx
    
    
    @staticmethod
    def bi_scaler(mtx : List[List[int]]) -> List[List[int]]:
        
        return mtx
    
    
    @staticmethod
    def adaptive_method(mtx : List[List[int]]) -> List[List[int]]:
        
        return mtx