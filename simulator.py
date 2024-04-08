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
    
    
class Simulator:
    
    m_mtx_field = List[List[int]]
    m_mtx_probe = List[List[int]]
    
    m_size : int = 20
    
    m_mu : float = 10.
    m_sigma : float = 2.
    
    m_strength : int = 10
    m_noise : int = 2
    
    m_probe_density : int = 10
    m_probe_sparsity : int = 1
    
    
    def __init__(self, size : int, mu : float, sigma : float,
                 strength : int = 10, noise : int = 3, density : int = 10, sparsity : int = 1,
                 type = MtxType.GAUSS_NORM):
        
        self.m_size = size
        self.m_mu = mu
        self.m_sigma = sigma
        self.m_strength = strength
        self.m_noise = noise
        self.m_probe_density = density
        self.m_probe_sparsity = sparsity
        
        match type:
            case MtxType.GAUSS_NORM:
                self.m_mtx_field = Simulator.get_gaussian_normal(size, strength)
            case MtxType.GAUSS_OFFS:
                self.m_mtx_field = Simulator.get_gaussian_offset(size, strength)
            case MtxType.RAD_GRAD_CLEAN:
                self.m_mtx_field = Simulator.get_radial_gradient_clean(size, strength)
            case MtxType.RAD_GRAD_NOISE:
                self.m_mtx_field = Simulator.get_radial_gradient_noise(size, strength)
        
        size_total = size ** 2
        probes_num = int(size_total * density / 100)
        
        self.m_mtx_probe = [[np.nan for x in range(size)] for y in range(size)]
        
        p_num = 0
        while p_num < probes_num:
            index = random.randrange(size_total)
            
            row = int(index / size)
            col = int(index % size)
            
            memo = set()
            if self.__backtrack(row, col, sparsity, memo) == False:
                continue
            
            self.m_mtx_probe[row][col] = self.m_mtx_field[row][col]
            p_num += 1
    
    
    def __backtrack(self, row : int, col : int, depth : int, memo : set) -> bool:
        
        if (row, col) in memo:
            return True
        
        if math.isnan(self.m_mtx_probe[row][col]) == False:
            return False
        else:
            memo.add((row, col))
        
        if depth == 0:
            return True
        
        result = True
        
        if row > 0:
            result &= self.__backtrack(row - 1, col, depth - 1, memo)
        if col > 0:
            result &= self.__backtrack(row, col - 1, depth - 1, memo)
        if row < self.m_size - 1:
            result &= self.__backtrack(row + 1, col, depth - 1, memo)
        if col < self.m_size - 1:
            result &= self.__backtrack(row, col + 1, depth - 1, memo)
            
        if row > 0 and col > 0:
            result &= self.__backtrack(row - 1, col - 1, depth - 1, memo)
        if row > 0 and col < self.m_size - 1:
            result &= self.__backtrack(row - 1, col + 1, depth - 1, memo)
        if row < self.m_size - 1 and col > 0:
            result &= self.__backtrack(row + 1, col - 1, depth - 1, memo)
        if row < self.m_size - 1 and col < self.m_size - 1:
            result &= self.__backtrack(row + 1, col + 1, depth - 1, memo)
            
        return result
    
            
    def get_field(self) -> List[List[int]]:
        return self.m_mtx_field
    
    
    def get_probe(self) -> List[List[int]]:
        return self.m_mtx_probe
            
    
    @staticmethod
    def get_gaussian_normal(size : int, mu : float, sigma : float) -> List[List[int]]:
        mtx = np.random.normal(mu, sigma, (size, size))
        return mtx
    
    
    @staticmethod
    def get_gaussian_offset(size : int, mu : float, sigma : float) -> List[List[int]]:
        mtx = np.random.normal(mu, sigma, (size, size))
        
        for col in range(size):
            for row in range(size):
                mtx[row][col] = math.trunc(mtx[row][col]) + mu
        
        return mtx
    
    
    @staticmethod
    def get_radial_gradient_clean(size : int, strength : int = 10) -> List[List[int]]:
        mtx = np.zeros((size, size), np.int32)
        
        center_x = size / 2
        center_y = center_x
        
        for x in range(size):
            for y in range(size):
                mtx[x][y] = int(strength - math.sqrt(pow(abs(center_x - x), 2) + pow(abs(center_y - y), 2)))
        
        return mtx
    
    
    @staticmethod
    def get_radial_gradient_noise(size : int, strength : int = 10, noise : int = 2) -> List[List[int]]:
        mtx = Simulator.get_radial_gradient_clean(size, strength)
        
        for x in range(size):
            for y in range(size):
                mtx[x][y] += random.randrange(-noise, noise + 1)
        
        return mtx
    
