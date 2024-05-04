import os, sys, cv2, tqdm
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
    ad_info['elements'] = {}
    json_file_name = f"{file_name.split('.')[0]}.json"
    json_path = f"{labels_dir}/{region_fol_name}/{json_file_name}"
    json_data = read_json(json_path)
    json_annotations = json_data[0]["annotations"]
    for annot in json_annotations:
      label, coordinates = annot['label'], annot["coordinates"]
      ad_info['elements'][label] = [int(coordinates["x"]), 
                                    int(coordinates["y"]), 
                                    int(coordinates["width"]), 
                                    int(coordinates["height"])]
    # plain layout sequence
    ad_info["plain_layout_seq"] = propotionality_scaling_of_coordinates(ad_info, prop_scaling_size).strip(" | ")
    all_data_with_constriants.append(ad_info)
    
  write_json("dataset/all_data_with_constriants.json", all_data_with_constriants)

  
if __name__ == "__main__":
  main()
  
  