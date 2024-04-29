import sys, os, re
sys.path.append(os.getcwd(), 'src')

import math
import numpy as np
import scipy as sp

def get_ele_size(bbox: list):
    # gets bbox's width and height
    return bbox[2], bbox[3] 

def get_ele_center(bbox: list):
    # return the center of boundary
    return bbox[0] + (bbox[2] // 2) , bbox[1] + (bbox[3] // 2)


class LayoutsCompare:
    def __init__(self, bboxes1, labels1, bboxes2, labels2):
        self.bboxes1, self.bboxes2 = bboxes1, bboxes2
        self.labels1, self.labels2 = labels1, labels2

    
    def get_shape_diff(self, bbox1, bbox2):
        # get the manhattan distance between centers of bbox
        x_center1, y_center1 = get_ele_center(bbox1)
        x_center2, y_center2 = get_ele_center(bbox2)
        
        return abs(x_center2 - x_center1) + abs(y_center1 - y_center2)
    
    
    def get_pos_diff(self, bbox1, bbox2):
        # get the euclidean distance between centers of bbox
        center1 = get_ele_center(bbox1)
        center2 = get_ele_center(bbox2)
        return sp.spatial.distance(center1, center2)

    
    def get_area_factor(self, bbox1, bbox2):
        # gets the area factor
        w_1, h_1 = get_ele_size(bbox1)
        w_2, h_2 = get_ele_size(bbox2)
        return math.sqrt(min(w_1*w_2, h_1*h_2))

    @property
    def get_ele_similarity(self, bbox1, label1, bbox2, label2):
        # gets the element similarity
        if label1 != label2: return 0
        
        pos_diff = self.get_pos_diff(bbox1, bbox2)
        shape_diff = self.get_shape_diff(bbox1, bbox2)
        area_factor = self.get_area_factor(bbox1, bbox2)
        return area_factor * pow(2, - pos_diff - 2*shape_diff)


    def get_layout_similarity(self):
        """Gets the similarity between two arbitary bboxes of two layouts

        Args:
            bboxes1 (_type_): _description_
            labels1 (_type_): _description_
            bboxes2 (_type_): _description_
            labels2 (_type_): _description_

        Returns:
            _type_: _description_
        """
        element_similarity = []
        for bbox1, label1 in zip(self.bboxes1, self.labels1):
            tmp_element_sim = []
            for bbox2, label2 in zip(self.bboxes2, self.labels2):
                tmp_element_sim.append(self.get_ele_similarity(bbox1, label1, bbox2, label2))
            element_similarity.append(tmp_element_sim)
            
        # Maximum weight matching   ????
        cost_matrix = np.array(element_similarity)
        row_idx, col_idx = sp.optimize.linear_sum_assignment(cost_matrix, maximize= True)
        return cost_matrix[row_idx, col_idx].sum()