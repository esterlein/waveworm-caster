import numpy as np
import math

class FieldCaster:
    
    m_simul_matrix = [[]]
    
    m_size = 20
    m_mu = 10
    m_sigma = 2
    
    def init_simul_matrix():
        m_simul_matrix = np.random.normal(m_mu, m_sigma, size(m_size, m_size))
        for col in range(m_size):
            for row in range(m_size):
                m_simul_matrix[row][col] = trunc(m_simul_matrix[row][col]) + m_mu
        return