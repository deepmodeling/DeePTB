
import numpy as np


# The Bloch class in Python defines a method to unfold k points based on a given Bloch factor.
class Bloch(object):

    def __init__(self,bloch_factor) -> None:
        
        if  isinstance(bloch_factor,list):
            bloch_factor = np.array(bloch_factor)
        
        assert bloch_factor.shape[0] == 3, "kpoint should be a 3D vector"
        self.bloch_factor = bloch_factor


    def unfold_points(self,k:list) -> np.ndarray:
        '''The `unfold_points` function generates expansion k points based on Bloch theorem and reshapes the
        output into a specific format.
        
        Parameters
        ----------
        k
            The `k` parameter in the `unfold_points` method represents the original k-point in the Brillouin zone.
        
        Returns
        -------
            The `unfold_points` method returns a reshaped array of expansion points calculated based on the
        input parameter `k` and the bloch factor `B`. The expansion points are created using B-casting rules
        and then reshaped into a 2D array with 3 columns.
        
        '''
                
        # check k is a 3D vector
        if isinstance(k,list):
            assert len(k) == 3, "kpoint should be a 3D vector"
        elif isinstance(k,np.ndarray):
            assert k.shape[0] == 3, "kpoint should be a 3D vector"
        else:
            raise ValueError("k should be a list or numpy array")

        # Create expansion points
        B = self.bloch_factor
        unfold = np.empty([B[2], B[1], B[0], 3])
        # Use B-casting rules (much simpler)
        unfold[:, :, :, 0] = (np.arange(B[0]).reshape(1, 1, -1) + k[0]) / B[0]
        unfold[:, :, :, 1] = (np.arange(B[1]).reshape(1, -1, 1) + k[1]) / B[1]
        unfold[:, :, :, 2] = (np.arange(B[2]).reshape(-1, 1, 1) + k[2]) / B[2]
        # Back-transform shape
        return unfold.reshape(-1, 3)