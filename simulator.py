import math
import random
import numpy as np

from enum import Enum


class MtxType(Enum):
    GAUSS_NORM = 1
    GAUSS_OFFS = 2
    RAD_GRAD_CLEAN = 3
    RAD_GRAD_NOISE = 4
    
    
class FieldSimulator:
    
    m_sim_mtx = [[]]
    
    m_size = 20
    
    m_mu = 10
    m_sigma = 2
    
    
    def __init__(self, size, mu, sigma, type = MtxType.GAUSS_NORM):
        self.m_size = size
        self.m_mu = mu
        self.m_sigma = sigma
        
        match type:
            case MtxType.GAUSS_NORM:
                self.init_mtx_gaussian_normal()
            case MtxType.GAUSS_OFFS:
                self.init_mtx_gaussian_offset()
            case MtxType.RAD_GRAD_CLEAN:
                self.init_radial_gradient_clean()
            case MtxType.RAD_GRAD_NOISE:
                self.init_radial_gradient_noise()
    
    
    def init_gaussian_normal(self):
        self.m_sim_mtx = np.random.normal(self.m_mu, self.m_sigma, (self.m_size, self.m_size))
    
    
    def init_gaussian_offset(self):
        self.m_sim_mtx = np.random.normal(self.m_mu, self.m_sigma, (self.m_size, self.m_size))
        for col in range(self.m_size):
            for row in range(self.m_size):
                self.m_sim_mtx[row][col] = math.trunc(self.m_sim_mtx[row][col]) + self.m_mu
    
    
    def init_radial_gradient_clean(self):
        self.m_sim_mtx = np.zeros((self.m_size, self.m_size), np.float64)
        
        center_x = self.m_size / 2
        center_y = self.m_size / 2
        
        for x in range(self.m_size):
            for y in range(self.m_size):
                self.m_sim_mtx[x][y] = math.sqrt(pow(abs(center_x - x), 2) + pow(abs(center_y - y), 2))
    
    
    def init_radial_gradient_noise(self):
        self.init_radial_gradient_clean()
        
        for x in range(self.m_size):
            for y in range(self.m_size):
                self.m_sim_mtx[x][y] += random.random()
        
        
                