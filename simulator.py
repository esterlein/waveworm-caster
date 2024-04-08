import math
import random
import numpy as np

from enum import Enum
from typing import List


class MtxType(Enum):
    
    GAUSS_NORM = 1
    GAUSS_OFFS = 2
    RAD_GRAD_CLEAN = 3
    RAD_GRAD_NOISE = 4
    
    
class FieldSimulator:
    
    m_mtx_field = List[List[float]]
    m_mtx_probe = List[List[float]]
    
    m_size : int = 20
    
    m_mu : float = 10.
    m_sigma : float = 2.
    
    m_strength : int = 100
    m_noise : int = 1
    
    m_probe_density : int = 10
    
    
    def __init__(self, size : int, mu : float, sigma : float,
                 strength : int = 100, noise : int = 1, density : int = 10,
                 type = MtxType.GAUSS_NORM):
        
        self.m_size = size
        self.m_mu = mu
        self.m_sigma = sigma
        self.m_strength = strength
        self.m_noise = noise
        self.m_probe_density = density
        
        match type:
            case MtxType.GAUSS_NORM:
                self.m_mtx_field = FieldSimulator.get_mtx_gaussian_normal(size, strength)
            case MtxType.GAUSS_OFFS:
                self.m_mtx_field = FieldSimulator.get_mtx_gaussian_offset(size, strength)
            case MtxType.RAD_GRAD_CLEAN:
                self.m_mtx_field = FieldSimulator.get_radial_gradient_clean(size, strength)
            case MtxType.RAD_GRAD_NOISE:
                self.m_mtx_field = FieldSimulator.get_radial_gradient_noise(size, strength)
        
        size_total = size ** 2     
        probes = int(size_total * density / 100)
        
        self.m_mtx_probe = [[np.nan for x in range(size)] for y in range(size)]
        
        for i in range(probes):
            index = random.randrange(size_total)
            x = int(index / size)
            y = int(index % size)
            self.m_mtx_probe[x][y] = self.m_mtx_field[x][y]
            
            
    def get_field(self) -> List[List[float]]:
        return self.m_mtx_field
    
    
    def get_probe(self) -> List[List[float]]:
        return self.m_mtx_probe
            
    
    @staticmethod
    def get_gaussian_normal(size : int, mu : float, sigma : float) -> List[List[float]]:
        mtx = np.random.normal(mu, sigma, (size, size))
        return mtx
    
    
    @staticmethod
    def get_gaussian_offset(size : int, mu : float, sigma : float) -> List[List[float]]:
        mtx = np.random.normal(mu, sigma, (size, size))
        
        for col in range(size):
            for row in range(size):
                mtx[row][col] = math.trunc(mtx[row][col]) + mu
        
        return mtx
    
    
    @staticmethod
    def get_radial_gradient_clean(size : int, strength : int = 100) -> List[List[float]]:
        mtx = np.zeros((size, size), np.float64)
        
        center_x = size / 2
        center_y = center_x
        
        for x in range(size):
            for y in range(size):
                mtx[x][y] = strength - math.sqrt(pow(abs(center_x - x), 2) + pow(abs(center_y - y), 2))
        
        return mtx
    
    
    @staticmethod
    def get_radial_gradient_noise(size : int, strength : int = 100) -> List[List[float]]:
        mtx = FieldSimulator.get_radial_gradient_clean(size, strength)
        
        for x in range(size):
            for y in range(size):
                mtx[x][y] += random.random()
        
        return mtx
    
