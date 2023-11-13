# custom_text_model.py
import autograd.numpy as np
from seldonian.models.models import SeldonianModel 

class CustomTextModel(SeldonianModel):
    def __init__(self):
        super().__init__()
    
    def predict(self,theta,data):
        """Data are lists of lists of characters
        Returns dot product of theta with each row of numerical
        representation of data. autograd doesn't like np.dot for some reason

        :param theta: The parameter weights
        :param data: A list of samples, where in this case samples are
            lists of length three with each element a single character
        """
        nums=[list(map(ord,x)) for x in data]
        return np.array([abs(theta[0]*row[0]+theta[1]*row[1]+theta[2]*row[2]) for row in nums])

