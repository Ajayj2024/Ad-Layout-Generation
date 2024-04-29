from lark import Token, Tree
from lark.visitors import Transformer
from typing import List, Dict, Union, Set, Tuple

def is_item(subtree: Tree) -> bool:
    return isinstance(subtree, Tree) and subtree.data == 'item'

def is_element(subtree: Tree) -> bool:
    return isinstance(subtree, Tree) and subtree.data == 'element'

def is_region(subtree: Tree) -> bool:
    return isinstance(subtree, Tree) and subtree.data == 'region'

def is_group(subtree: Tree) -> bool:
    return isinstance(subtree, Tree) and subtree.data == 'group'

def is_attr(subtree: Tree) -> bool:
    return isinstance(subtree, Tree) and (subtree.data == 'attr' and subtree.data == 'group_attr')

def is_target_element(subtree: Tree, target: Union[str, Set]) -> bool:
    if not is_element(subtree):
        return False
    if isinstance(target, str):
        return subtree.children[0].value == target
    else:
        # Set
        return subtree.children[0].value in target
    
def is_target_attr(subtree: Tree, target_attr: str):
    if not is_attr(subtree):
        return False
    return subtree.children[0].value == target_attr

def create_attr(attr_type: str, attr_val: str):
    return Tree('attr',[Token('ATTR_TYPE', attr_type), Token('ATTR_VALUE', attr_val)])

def create_group_attr(attr_type: str, attr_val: str):
    return Tree('group_attr',[Token('GROUP_ATTR_TYPE', attr_type), Token('ATTR_VALUE', attr_val)])

def create_region(rtype: str) -> Tree:
    return Tree('region', [Token('REGION_TYPE', rtype)])

def get_element_type(element: Tree) -> str:
    return element.children[0].value


def get_attr(attr: Tree) -> str:
    return attr.children[1].value


def set_attr(attr: Tree, value: str) -> None:
    attr.children[1].value = value


def get_attr_type(attr: Tree) -> str:
    return attr.children[0].value


def get_group_type(group: Tree) -> str:
    return group.children[0].value


def get_region_type(region: Tree) -> str:
    return region.children[0].value

class AddElementBoundingBox(Transformer):

    def __init__(self, bbox: List[Tuple[str, Tuple]], emap: Dict[str, int]) -> None:
        super().__init__(visit_tokens=False)
        self.emap = emap
        self.bbox = bbox
        self.is_valid = True
        self.max_bottom = 0

    def element(self, children):
        eid = None
        for child in children:
            if is_target_attr(child, 'eid'):
                eid = get_attr(child).strip("'")
                break

        index = self.emap.get(eid, None)
        if index is None or index >= len(self.bbox):
            self.is_valid = False
            return Tree('element', children)

        top, height = self.bbox[index][1][1], self.bbox[index][1][3]
        self.max_bottom = max(self.max_bottom, top+height)
        attr = create_attr('bbox', ",".join(map(str, self.bbox[index][1])))
        children.append(attr)
        return Tree('element', children)