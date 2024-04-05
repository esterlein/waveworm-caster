import math
import numpy as np

class FieldCaster:
    
    m_simul_matrix = [[]]
    
    m_size = 20
    m_mu = 10
    m_sigma = 2
    
    def init_mtx_gaussian_normal(self):
        m_simul_matrix = np.random.normal(self.m_mu, self.m_sigma, (self.m_size, self.m_size))
        return
    
    def init_mtx_gaussian_offset(self):
        m_simul_matrix = np.random.normal(self.m_mu, self.m_sigma, (self.m_size, self.m_size))
        for col in range(self.m_size):
            for row in range(self.m_size):
                m_simul_matrix[row][col] = math.trunc(m_simul_matrix[row][col]) + self.m_mu
        return