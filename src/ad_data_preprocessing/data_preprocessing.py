
def propotionality_scaling_of_coordinates(ad_info: dict, new_img_size: list):
    org_width, org_height = ad_info['canvas_width'], ad_info['canvas_height']
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
        ele = temp['type']
        x, y, w, h = temp['position'] 
       
        
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


if __name__ == "__main__":
    ad_info = {
        "canvas_width": 1440,
        "canvas_height": 2560,
    "elements": [
      {
        "type": "text button",
        "position": [
          70,
          312,
          1300,
          105
        ]
      },
      {
        "type": "text button",
        "position": [
          35,
          109,
          280,
          125
        ]
      },
      {
        "type": "text",
        "position": [
          0,
          129,
          1440,
          85
        ]
      },
      {
        "type": "text",
        "position": [
          88,
          540,
          1264,
          1491
          ]
        }
      ]
    }
    res = propotionality_scaling_of_coordinates(ad_info, [144, 256])
    res = res.strip(" ").strip("|")
    print(res)