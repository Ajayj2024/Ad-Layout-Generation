import sys, os, re
sys.path.append(os.path.join(os.getcwd(), 'src'))

from lark import Token
from lark import Tree as LarkTree
from lark.visitors import Transformer
from dataclasses import dataclass

# from utils.exception import CustomeException
# from utils.logger import logging
from config.config import MAP_CONFIG, CONFIG
from ir.ir_utils import *
from ir.parser import parse_grammar

map_config = MAP_CONFIG.mapping

def is_num(val: str):
    return re.match(r'\d+', val)


class GroupExpansion:
    pass

class AddElementId(Transformer):
    def __init__(self, visit_token: bool = True) -> None:
        super().__init__(visit_token)
        self.id = 0
    
    def add_ele_id(self, children):
        attr = create_attr('eid', f"e_{self.id}")
        children.append(attr)
        self.id += 1
        return Tree('element', children)
    
    
# mapping and Sorting the information in order ele_type ele_pos ele_size 
@dataclass
class ElementMapping:
    ele_type: str
    # ele_id: str # ???
    ele_pos: str
    ele_size: str
    repeat_num: int
    element_map = map_config['ELEMENT_MAP_TYPE']
    pos_priority = map_config["POSITION_PRIORITY"]
    pos_map = map_config['POSITION_MAP']
    size_priority = map_config['SIZE_PRIORITY']
    def __post_init__(self):
        self.post_ele_type = self.ele_type
        
        # self.post_ele_type = self.element_map[self.ele_type]
        self.post_pos = self.pos_map[self.ele_pos]
        self.post_size_priority = self.size_priority[self.ele_size]
        self.post_pos_priority = self.pos_priority[self.ele_pos]
        
    def get_elements_lst(self):
        # return [f"{self.post_ele_type} {self.post_pos} {self.post_size}"] * self.repeat_num
        return [f"{self.post_ele_type} {self.post_pos} {self.ele_size}"] * self.repeat_num
    
    @staticmethod
    def sort_elements(elements: List[str]):
        elements.sort(key = lambda e: (e.post_ele_type, e.post_pos_priority, e.post_size_priority))
        
        
        
class PlacementIR:
    UNDEFINED = 'undefined'
    def order_elements(self, element: Tree):
        element_type = element.children[0].value
        position, size, repeat_num = self.UNDEFINED, self.UNDEFINED, 1
        for child in element.children:
            if isinstance(child, Token) and child.type == 'ELEMENT_TYPE':
                element_type = child

            if isinstance(child, Tree) and child.data == 'attr':
                attr_child_lst = child.children
                attr_type, attr_value = attr_child_lst 
                if attr_type == 'position':
                    position = attr_value.strip("'")

                elif attr_type == 'repeat':
                    repeat_num = int(attr_value.strip("'"))

                elif attr_type == 'size':
                    size = attr_value.strip("'")
                # ?? 
                # elif attr_type == 'eid':
                #     eid = attr_value

        # print(element_type, position, size)
        # print([f"{element} {position} {size}"]*repeat_num)
        element_mapping = ElementMapping(ele_type= element_type, 
                                         ele_pos= position, 
                                         ele_size= size, 
                                         repeat_num= repeat_num)
        
        return element_mapping
    
    def order_groups(self):
        pass
    
    def order_items(self):
        pass
    
    def __call__(self, region_tree: Tree):
        region_type, elements = None, []
        for child in region_tree.children:
            if isinstance(child, Token) and child.type == 'REGION_TYPE':
                region_type = child

            elif isinstance(child, Tree) and child.data == 'element':
                elements.append(self.order_elements(child))

            ElementMapping.sort_elements(elements)
        
        constraints = []
        for ele in elements:
            constraints.extend(ele.get_elements_lst())
        return f"{region_type} : " + ' | '.join(constraints)
        
        
class ConstraintExecutor:
    def __init__(self, grammar_file_path) -> None:
        self.parser = parse_grammar(grammar_file_path= grammar_file_path)
        self.placement_fn = PlacementIR()
        
    def get_constraints(self, ir):
        parse_tree = self.parser.parse(ir)
        return self.placement_fn(parse_tree.children[0])
    
    
# if __name__ == "__main__":
#     grammar_file_path = CONFIG.params['DATASET']['grammar_file_path']
#     executor = ConstraintExecutor(grammar_file_path= grammar_file_path)
#     input_ir = "[region:ElectronicDevice[el:logo [attr:position'left']] [el:text [attr:position'left']] [el:image] [el:price] [el:icon [attr:size'large'] [attr:repeat'3']]]"
#     print(executor.get_constraints(input_ir))
#     # print(ElementMapping(ele_type= 'element_type', ele_pos= 'position', ele_size= 'size', repeat_num= 3))