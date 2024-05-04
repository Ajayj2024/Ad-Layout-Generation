import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

import re
from typing import List, Dict, Tuple
from utils.exception import CustomeException
from utils.logger import logging

# Text Preprocessing
class TextPreprocessor:
    VALUE_PLACEHOLDER = 'value_'
    def __init__(self, replace_value: bool = False) -> None:
        self.replace_value = replace_value
        
    def preprocessor(self, text: str) -> Tuple[str, dict]:
        '''
        example: '#He is  #   2good "op", John"s "ghui"'
        returns: ('he is 2good "value_0", john"value_1"ghui"', {'value_0': 'op', 'value_1': 's'})
        '''
        # replace # to "", strip white space, lower
        result = text.replace("#", "").strip().lower()
        result = result.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")
        
        # replaces one or more space occurences to one space occurence
        result = re.sub("\s+", " ", result)
        value_map = None
        if self.replace_value:
            result, value_map = self.extract_explicit_values(result)
    
        return result, value_map
    
    
    def extract_explicit_values(self, text: str) -> Tuple:
        
        result = re.sub(r"(\w)'s\s+", r'\g<1>`s ', text)
        value_map = dict()
        values = re.findall(r'".*?"', result)
        values_1 = re.findall(r"'.*?'", result)
        if len(values_1) == 1 and any([punct in values_1[0] for punct in [",", "."]]):
            values_1 = list()
        values.extend(values_1)
        for vidx, v in enumerate(values):
            placeholder = f'{self.VALUE_PLACEHOLDER}{vidx}'
            value_map[placeholder] = v.strip('"').strip()
            result = result.replace(v, f'"{placeholder}"', 1) # replace one occurrence per time
        result = re.sub(r"(\w)`s\s+", r"\g<1>'s ", result)
        result = re.sub("\s+", " ", result)
        
        return result, value_map
    
# Preprocesses IR
class IRProcessor:
    TAG_RENAMES = [
        ("gp:navgroups", "gp:navgroup"),
        ("region:singleinfo", "region:info"),
        ("gattr:", "group_prop:"),
        ("attr:", "prop:"),
        ("gp:", "group:"),
        ("el:", "element:"),
    ]
    def __init__(self, remove_value: bool = True, replace_value: bool = False) -> None:
        self.remove_value = remove_value
        self.replace_value = replace_value
        
        
    def preprocess(self, ir: str, idx2value: dict = None) -> str:
        
        result = ir.replace("[", " [ ").replace("]", " ] ").strip().lower()
        result = re.sub(r"attr:(.*?)'", r"attr:\g<1> '", result)

        if self.remove_value:
            result = re.sub(r"\[\s*attr:value\s*'.*?'\s*\]", r"", result)
            result = re.sub(r"\[\s*gattr:names\s*'.*?'\s*\]", r"", result)

        for tag, new_tag in self.TAG_RENAMES:
            result = result.replace(tag, new_tag)

        result = result.replace(':', ' : ')
        result = re.sub("\s+", " ", result)

        if not self.remove_value and self.replace_value and idx2value is not None:
            value_map = {
                v.replace("`", "").replace("'", ""): k
                for k, v in idx2value.items()
            }
            result = self.replace_explicit_values(result, value_map)

        return result
    
    
    def replace_explicit_values(self, ir: str, value_map: dict) -> str:
        # prop
        result = ir
        value_attrs = re.findall(r"(\[ prop:value '(.*?)' \])", ir)
        for attr, value in value_attrs:
            _value = value.strip()
            if _value in value_map:
                result = result.replace(
                    attr, f"[ prop:value '{value_map[_value]}' ]")
            else:
                # strip
                _value = ", ".join([_v.strip() for _v in _value.split(",")])
                result = result.replace(attr, f"[ prop:value '{_value}' ]")

        # names
        name_attrs = re.findall(r"(\[ group_prop:names '(.*?)' \])", ir)
        for attr, value in name_attrs:
            names = [n.strip() for n in value.split(",")]
            new_names = list()
            for name in names:
                if name in value_map:
                    new_names.append(value_map[name])
                else:
                    new_names.append(name)
            new_names = ", ".join(new_names)
            result = result.replace(attr, f"[ group_prop:names '{new_names}' ]")
        return result

    def postprocess(self,
                    ir: str,
                    remove_attrs: bool = False,
                    recover_labels: bool = False,
                    recover_values: bool = False,
                    value_map: dict = None) -> str:
        
        result = ir.replace("[", " [ ").replace("]", " ] ").strip().lower()
        result = result.replace(" : ", ":")
        result = re.sub(r"attr:(.*?)'", r"prop:\g<1> '", result)
        if remove_attrs:
            result = re.sub(r"\[\s*attr:(value|size|position|repeat)\s*'.*?'\s*\]", r"", result)
            
        if recover_labels:
            for tag, new_tag in self.TAG_RENAMES[1:]:
                result = result.replace(new_tag, tag)
            for tag, new_tag in [("region:singleinfo", "region:SingleInfo")]:
                result = result.replace(tag, new_tag)
            result = re.sub(r'\s*\[\s*', '[', result)
            result = re.sub(r'\s*\]\s*', ']', result)
            if recover_values and value_map is not None:
                for idx, value in value_map.items():
                    _value = value.replace("'", "")
                    result = result.replace(f"'{idx}'", f"'{_value}'")
                    result = result.replace(f" {idx},", f" {_value},")
                    result = result.replace(f"'{idx},", f"'{_value},")
                    result = result.replace(f" {idx}'", f" {_value}'")
                # replace &
                result = result.replace("&", " and ")
        result = re.sub("\s+", " ", result)
        return result
    
# if __name__ == "__main__":
#     # tp = TextPreprocessor(replace_value= True)
#     # text = "########,*^this page is a 'mobile camera' cutie shooting process. on the top of the page, there are two images, one on the left and the other on the far right. at the bottom of the page, there are three images, one on the left, one in the middle, and the else on the far right, for which users can choose different types of cutie shooting styles."
#     # print(tp.preprocessor(text))
    # tp = IRProcessor(True)
    # ir = "[region:SingleInfo [el:image [attr:position'top'] [attr:repeat'2'] ] [el:image [attr:position'bottom'] [attr:repeat'3'] ] ]"
    # print(tp.postprocess(ir))