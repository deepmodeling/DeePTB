
import numpy as np


class Bloch(object):
    def __init__(self,bloch_factor) -> None:
        
        if  isinstance(bloch_factor,list):
            bloch_factor = np.array(bloch_factor)
        
        assert bloch_factor.shape[0] == 3, "kpoint should be a 3D vector"
        self.bloch_factor = bloch_factor


    def unfold_points(self,k):
                
    
        # Create expansion points
        B = self.bloch_factor
        unfold = np.empty([B[2], B[1], B[0], 3])
        # Use B-casting rules (much simpler)
        unfold[:, :, :, 0] = (np.arange(B[0]).reshape(1, 1, -1) + k[0]) / B[0]
        unfold[:, :, :, 1] = (np.arange(B[1]).reshape(1, -1, 1) + k[1]) / B[1]
        unfold[:, :, :, 2] = (np.arange(B[2]).reshape(-1, 1, 1) + k[2]) / B[2]
        # Back-transform shape
        return unfold.reshape(-1, 3)