import sys, os 
sys.path.append(os.path.join(os.getcwd(), 'src'))
import re
from file_utils import read_json,makedirs
from config.dictionary import LABEL2COLORS, CANVAS_SIZE
import cv2
import matplotlib.colors as mcolors
import numpy as np
from tqdm import tqdm
# Function to convert color name to RGB tuple
def color_name_to_rgb(color_name):
    # Get RGB tuple from color name
    rgb_tuple = mcolors.CSS4_COLORS.get(color_name)
    if rgb_tuple:
        return mcolors.to_rgb(rgb_tuple)
    else:
        raise ValueError("Invalid color name")

def draw_rectangle(img, start_pt, end_pt, label):
    center = (np.array(start_pt) + np.array(end_pt)) // 2
    color_name = LABEL2COLORS[label]
    color_code = [int(255*code) for code in color_name_to_rgb(color_name) ]     # give code in RGB format
    color_code_bgr = tuple(reversed(color_code))
    cv2.rectangle(img, start_pt, end_pt, color_code_bgr, thickness= 2)
    cv2.putText(img, label, center.tolist(), cv2.FONT_HERSHEY_SIMPLEX,  1, color_code_bgr, 1, cv2.LINE_AA)
def visualizer(pred_list: list,path):
    j = 1
    for out in pred_list:
        pattern = r'^[^a-zA-Z0-9]*|[^a-zA-Z0-9]*$'
        # Remove non-alphanumeric characters from the beginning and end of the text
        out = re.sub(pattern, '', out)
        out = out.replace(' | ', '|')
        layout = np.ones((720, 720, 3), dtype=np.uint8) * 255
        label_pos = out.split('|')
        for lb in label_pos:
            lb = lb.strip(" ")
            spacing_count = 0
            for i in range(-1,-len(lb)-1, -1):
                if lb[i] == " ":
                    spacing_count += 1
                    
                if spacing_count == 4:
                    label = lb[:len(lb)+i].strip()
                    print(lb[len(lb) + i + 1:].strip().split(' '))
                    xc, yc, w, h = [int(c) for c in lb[len(lb) + i + 1:].strip().split(' ')]
                    break
            start_pt = [xc - (w//2), yc - (h//2)]
            end_pt = [xc + (w//2), yc + (h//2)]
            draw_rectangle(layout, start_pt, end_pt, label) 
        cv2.imwrite(f"{path}/{j}.png", layout)
        print("image")
        j += 1
                    
            
                    
                
        
        
if __name__ == "__main__":
    out_data = read_json('predictions/place_stage/prediction.json')
    i = 1
    for data in tqdm(out_data):
        out_list = data["output"]
        makedirs(f"layouts/{i}")
        print(out_list)
        visualizer(out_list, f"layouts/{i}")
        i += 1
        
            
   