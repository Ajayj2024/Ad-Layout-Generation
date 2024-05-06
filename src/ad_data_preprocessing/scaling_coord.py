
def propotionality_scaling_of_coordinates(ad_info: dict, new_img_size: list):
    org_width, org_height = ad_info['ad_width'], ad_info['ad_height']
    new_width, new_height = new_img_size
    elements = ad_info['elements']
    explicit_constraint = ""
    
    aspect_ratio = org_width / org_height

    if aspect_ratio > 1:
        scaling_factor = new_width / org_width
    else:
        scaling_factor = new_height / org_height
        
    scaled_image_width = int(org_width * scaling_factor)
    scaled_image_height = int(org_height * scaling_factor)
    for temp in elements:
        ele = temp["ele_type"]
        x, y, w, h = temp["coordinates"]
       
        
        scaled_x = int(x * scaling_factor)
        scaled_y = int(y * scaling_factor)
        scaled_w = int(w * scaling_factor)
        scaled_h = int(h * scaling_factor)
        
        if scaled_x + scaled_w > scaled_image_width:
            scaled_w = scaled_image_width - scaled_x
            
        if scaled_y + scaled_h > scaled_image_height:
            scaled_h = scaled_image_height - scaled_y
            
        explicit_constraint += f"{ele} {scaled_x} {scaled_y} {scaled_w} {scaled_h} | "
        
        
    return explicit_constraint

def org_coordinates():
  pass

