import os, sys, cv2, tqdm
from collections import defaultdict
sys.path.append(os.path.join(os.getcwd(),'src'))

from scaling_coord import *
from ir.executor import ConstraintExecutor
from config.config import CONFIG
from utils.file_utils import read_json, write_json

prop_scaling_size = [720, 720]
config = CONFIG.params
ads_dir = 'Layout Generation Data/Ads'
labels_dir = 'Layout Generation Data/labels'
REGIONS = ['Electronic Device', 'Food']
REGION2FOLDER = {k: v for k,v in zip(REGIONS, os.listdir(ads_dir))}
def IR2Constraints(ir: str):
  constraint_excutor = ConstraintExecutor(grammar_file_path= config['DATASET']['grammar_file_path'])
  return constraint_excutor.get_constraints(ir)

def find_size(img_path):
  try:
    img = cv2.imread(img_path)
    w, h = img.shape[1], img.shape[0]
    return w, h

  except:
    print(img_path)
    
def map_label2coord(labels_pos_lst):
  labels2coord = defaultdict(list)
  for lab in labels_pos_lst:
    space_count = 0
    for i in range(-1,-len(lab)-1, -1):
      if lab[i] == " ":
        space_count += 1
        
      if space_count == 4:
        split_idx = len(lab) + i
        # mapping label with coordinate points
        labels2coord[lab[:split_idx]].append([int(val) for val in lab[split_idx+1:].split(" ")])
        break
      
  return labels2coord


def map_label2attr(labels_attr_lst):
  labels2attr = defaultdict(list)
  for lab in labels_attr_lst:
    space_count = 0
    for i in range(-1,-len(lab)-1, -1):
      if lab[i] == " ":
        space_count += 1
        
      if space_count == 2:
        split_idx = len(lab) + i
        # mapping label with coordinate points
        labels2attr[lab[:split_idx]].append(lab[split_idx+1:].split(" "))
        break
      
  return labels2attr

def get_top_coord(coord):
  # print(coord)
  top_coord = None
  min_dist = float('inf')
  for pt in coord:
    # print(pt)
    if pt[1] < min_dist: min_dist, top_coord = pt[1], pt
  print(top_coord)
  coord.remove(top_coord)
  return top_coord

def get_buttom_coord(coord):
  buttom_coord = None
  max_dist = float('-inf')
  for pt in coord:
    if pt[1] > max_dist: max_dist, bottom_coord = pt[1], pt
    
  coord.remove(bottom_coord)
  return bottom_coord

def get_right_coord(coord):
  right_coord = None
  max_dist = float('-inf')
  for pt in coord:
    if pt[0] > max_dist: max_dist, right_coord = pt[1], pt
    
  coord.remove(right_coord)
  return right_coord

def get_left_coord(coord):
  left_coord = None
  min_dist = float('inf')
  for pt in coord:
    if pt[0] < min_dist: min_dist, left_coord = pt[1], pt
    
  coord.remove(left_coord)
  return left_coord

def get_small_coord(coord):
  small_coord = None
  min_area = float('inf')
  for pt in coord:
    if pt[2] * pt[3] < min_area:
      min_area = pt[2] * pt[3]
      small_coord = pt
      
  coord.remove(pt)
  return small_coord

def get_large_coord(coord):
  large_coord = None
  max_area = float('-inf')
  for pt in coord:
    if pt[2] * pt[3] > max_area:
      max_area = pt[2] * pt[3]
      large_coord = pt
      
  coord.remove(pt)
  return large_coord

def sort_label_position(ad_info):
  sorted_seq = ""
  labels_pos_split = [l.strip() for l in ad_info['plain_layout_seq'].split('|')]
  label2coord = map_label2coord(labels_pos_split)
  sorted_labels = sorted(label2coord.keys())
  for sl in sorted_labels:
    for pos in label2coord[sl]:
      sorted_seq += f"{sl} {pos[0]} {pos[1]} {pos[1]} {pos[1]} | "
      
  return sorted_seq
  
def rearrange_label_position(ad_info):
  labels_pos_split = [l.strip() for l in ad_info['plain_layout_seq'].split('|')]
  ## mapping labels to coordinates
  labels2coord = map_label2coord(labels_pos_split)
  print(labels2coord)
  ## mappping serialization constraint's elements to attributes
  region_type, label_pos_const = [l.strip() for l in ad_info['serialize_constraints'].split(':')]
  labels_attr_split = [l.strip() for l in label_pos_const.split('|')]
  labels2attr = map_label2attr(labels_attr_split)
    
  sorted_label_pos = ""
  for lab in labels2attr.keys():
      if len(labels2coord[lab]) == 1:
        sorted_label_pos += f"{lab} "
        for x in labels2coord[lab][0]: 
          sorted_label_pos += f"{x} " 
        sorted_label_pos += "| "
      elif len(labels2coord[lab]) > 1:
        attributes, coordinates = labels2attr[lab], labels2coord[lab]
        positions, sizes = [i[0] for i in attributes], [i[1] for i in attributes]
        ptr = 0
        
        
        while ptr != len(positions):
          sorted_label_pos += f"{lab} "
          if ptr+1 == len(positions) or positions[ptr] != positions[ptr+1]:
            if positions[ptr] == 'top': coord = get_top_coord(coordinates)
              
            elif positions[ptr] == 'bottom': coord = get_buttom_coord(coordinates)
            elif positions[ptr] == 'left': coord = get_left_coord(coordinates)
            elif positions[ptr] == 'right': coord = get_right_coord(coordinates)
            
            for x in coord: sorted_label_pos += f"{x} "
            sorted_label_pos += "| "
            
          else:
            if sizes[ptr] == 'small': coord = get_small_coord(coordinates) 
            elif sizes[ptr] == 'large': coord = get_large_coord(coordinates) 
            else: 
              coord = coordinates[0]
              coordinates.remove(coord)
            for x in coord: sorted_label_pos += f"{x} "
            
            sorted_label_pos += "| "
          ptr += 1
  return sorted_label_pos



def main():
  all_data = read_json('dataset/all_data.json')
  all_data_with_constriants = []
  for ad_info in tqdm.tqdm(all_data):
    region_type = ad_info['region']
    region_fol_name = REGION2FOLDER[region_type]
    file_name = ad_info['file_name']
    
    # find canvas size
    img_path = f"{ads_dir}/{region_fol_name}/{file_name}"
    width, height = find_size(img_path)
    ad_info["ad_width"], ad_info["ad_height"] = width, height
    
    # convert ir to constraint execution
    ad_info['serialize_constraints'] = IR2Constraints(ad_info['ir'].lower())
    
    # Add elements coordinates
    ad_info['elements'] = []
    json_file_name = f"{file_name.split('.')[0]}.json"
    json_path = f"{labels_dir}/{region_fol_name}/{json_file_name}"
    json_data = read_json(json_path)
    json_annotations = json_data[0]["annotations"]
    for annot in json_annotations:
      label, coordinates = annot['label'], annot["coordinates"]
      d = {
        "ele_type": label,
        "coordinates": [int(coordinates["x"]), int(coordinates["y"]), int(coordinates["width"]), int(coordinates["height"])]
      }
      ad_info['elements'].append(d)
    # print(ad_info['elements'])
    # plain layout sequence converting the coordinates to propotionality scaling
    ad_info["plain_layout_seq"] = propotionality_scaling_of_coordinates(ad_info, prop_scaling_size).strip(" | ")
    
    # sorting the plain_layout_sequence according to serilize constraints
    
    ad_info["plain_layout_seq"] = sort_label_position(ad_info)
    all_data_with_constriants.append(ad_info)
    
  write_json("dataset/all_data_with_constriants.json", all_data_with_constriants)

  
if __name__ == "__main__":
  main()
  
  