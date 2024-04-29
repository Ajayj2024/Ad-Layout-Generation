from dataclasses import dataclass

@dataclass
class Region:
    # Region: region_type, region's width and height, region's element
    rtype: str  # Region Type
    width: int  
    height: int
    elements: list
    
    
@dataclass
class ElementStyle:
    # font style
    font_size: int = None
    font_style: str = None
    text_align: str = None
    text_vertical_align: str = None
    text_decoration: str = None
    text_color: str = None
    
    # Font Alignment
    TEXT_LEFT_ALIGNMENT: str = 'left'
    TEXT_RIGHT_ALIGNMENT: str = 'right'
    TEXT_CENTER_ALIGNMENT: str = 'center'

    TEXT_VERTICAL_TOP_ALIGNMENT: str = 'top'
    TEXT_VERTICAL_BOTTOM_ALIGNMENT: str = 'bottom'
    TEXT_VERTICAL_CENTER_ALIGNMENT: str = 'middle'
    
    TEXT_DECORATION_UNDERLINE: str = 'underline'

    # Border Style
    border: str = None
    border_weight: int = 0
    border_color: str = None
    
    
@dataclass
class Elements:
    x: int ; y: int ; width: int ; height: int ; etype: str
    text: str = None
    image: str = None
    element_style: ElementStyle = None
    
    def post_init(self):
        if self.element_style is None:
            self.element_style = ElementStyle()
        
        # Update default style according to element type
        # if self.style.text_align is None:
        #     if self.etype in {"button"}:
        #         self.style.text_align = ElementStyle.TEXT_CENTRE_ALIGN
        #     else:
        #         self.style.text_align = ElementStyle.TEXT_LEFT_ALIGN
        # if self.style.text_vertical_align is None:
        #     if self.etype in {"button"}:
        #         self.style.text_vertical_align = ElementStyle.TEXT_VERTICAL_MIDDLE_ALIGN
        #     else:
        #         self.style.text_vertical_align = ElementStyle.TEXT_VERTICAL_TOP_ALIGN

            
    def get_ltwh_bbox(self):
        return [self.x, self.y, self.width, self.height]
    
    def get_ltrb_bbox(self):
        right, bottom = self.x + self.width, self.y + self.height
        return [self.x, self.y, right, bottom]
    
    # def has_image(self):
    #     return self.image is not None and isinstance(self.image, str) and self.image.startswith("https:")

    # def has_text(self):
    #     return self.text is not None and isinstance(self.text, str) and len(self.text) > 0

def read_region_bbox():
    pass
    