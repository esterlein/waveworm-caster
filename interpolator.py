import math
import numpy as np

from enum import Enum
from typing import List

from fancyimpute import (
    KNN,
    NuclearNormMinimization,
    SoftImpute,
    IterativeImputer,
    IterativeSVD,
    BiScaler
)


class DataRange(Enum):
    NORMALIZED = 1
    ADC_BYTE = 255


class InterpolationType(Enum):
    
    KNN = 1
    NNM = 2
    SOFT_THOLD_SVD = 3
    ITERATIVE_ROUND = 4
    ITERATIVE_SVD = 5
    MATRIX_FACTORIZATION = 6
    BI_SCALER = 6
    ADAPTIVE_METHOD = 7
    

class Interpolator:
    
    m_mtx_probe = List[List[int]]
    m_mtx_inter = List[List[int]]
    
    def __init__(self, probe : List[List[int]], type = InterpolationType.NNM):
        
        self.m_mtx_probe = probe
        
        match type:
            case InterpolationType.KNN:
                self.m_mtx_inter = Interpolator.k_nearest_neighbors(probe)
            case InterpolationType.NNM:
                self.m_mtx_inter = Interpolator.nuclear_norm_minimization(probe)
            case InterpolationType.SOFT_THOLD_SVD:
                self.m_mtx_inter = Interpolator.soft_thold_single_val_decomp(probe)
            case InterpolationType.ITERATIVE_ROUND:
                self.m_mtx_inter = Interpolator.iterative_round_impute(probe)
            case InterpolationType.ITERATIVE_SVD:
                self.m_mtx_inter = Interpolator.iterative_single_val_decomp(probe)
            case InterpolationType.MATRIX_FACTORIZATION:
                self.m_mtx_inter = Interpolator.factorization(probe)
            case InterpolationType.BI_SCALER:
                self.m_mtx_inter = Interpolator.bi_scaler(probe)
            case InterpolationType.ADAPTIVE_METHOD:
                self.m_mtx_inter = Interpolator.adaptive_method(probe)
        
        
    def get_interpolated(self) -> List[List[int]]:
        return self.m_mtx_inter
    
    
    @staticmethod
    def k_nearest_neighbors(probe : List[List[int]], k : int = 2, orientation = "rows") -> List[List[int]]:
        mtx = KNN(k, orientation).fit_transform(probe)
        return mtx
    
    
    @staticmethod
    def nuclear_norm_minimization(probe : List[List[int]]) -> List[List[int]]:
        mtx = NuclearNormMinimization.fit_transform(probe)
        return mtx
    
    
    @staticmethod
    def soft_thold_single_val_decomp(probe : List[List[int]], shrinkage_value : int = 25) -> List[List[int]]:
        mtx = SoftImpute(shrinkage_value).fit_transform(probe)
        return mtx
    
    
    @staticmethod
    def iterative_round_impute(probe : List[List[int]], nearest_features : int = 80, max_iter : int = 50) -> List[List[int]]:
        mtx = IterativeImputer(nearest_features, max_iter).fit_transform(probe)
        return mtx
    
    
    @staticmethod
    def iterative_single_val_decomp(probe : List[List[int]], rank : int = 10, init_fill = "nan") -> List[List[int]]:
        mtx = IterativeSVD(rank, init_fill).fit_transform(probe)
        return mtx
    
    
    @staticmethod
    def factorization(probe : List[List[int]]) -> List[List[int]]:
        return
    
    
    @staticmethod
    def bi_scaler(probe : List[List[int]]) -> List[List[int]]:
        
        return
    
    
    @staticmethod
    def adaptive_method(probe : List[List[int]]) -> List[List[int]]:
        
        return