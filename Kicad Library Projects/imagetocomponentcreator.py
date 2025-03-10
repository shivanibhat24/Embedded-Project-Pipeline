import cv2
import numpy as np
import os
import sys
import json
from PIL import Image
import pytesseract
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt
import pcbnew
import re

class KiCadComponentGenerator:
    def __init__(self, config_path="config.json"):
        """Initialize the KiCad component generator with configuration."""
        self.config = self.load_config(config_path)
        self.setup_paths()
    
    def load_config(self, config_path):
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file not found at {config_path}. Using default configuration.")
            return {
                "output_dir": "output",
                "temp_dir": "temp",
                "tesseract_path": "/usr/bin/tesseract",
                "kicad_lib_path": os.path.expanduser("~/Documents/KiCad/libraries"),
                "pin_mapping_database": "pin_database.json"
            }
    
    def setup_paths(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(self.config["output_dir"], exist_ok=True)
        os.makedirs(self.config["temp_dir"], exist_ok=True)
        
        # Set Tesseract path if provided in config
        if "tesseract_path" in self.config:
            pytesseract.pytesseract.tesseract_cmd = self.config["tesseract_path"]
    
    def process_image(self, image_path):
        """Process sensor image to detect shape, pins, and features."""
        print(f"Processing sensor image: {image_path}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Preprocess image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours to detect component outline
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("No contours found in the image")
        
        # Find the largest contour (presumably the sensor)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the component shape
        epsilon = 0.02 * cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        
        # Determine component shape
        self.component_dims = self._get_component_dimensions(approx)
        self.component_shape = self._determine_component_shape(approx)
        
        # Extract text from image for potential part number identification
        extracted_text = pytesseract.image_to_string(Image.fromarray(gray))
        self.part_info = self._extract_part_info(extracted_text)
        
        return {
            "shape": self.component_shape,
            "dimensions": self.component_dims,
            "part_info": self.part_info
        }
    
    def process_pin_diagram(self, pin_diagram_path):
        """Process pin diagram image to extract pin information."""
        print(f"Processing pin diagram: {pin_diagram_path}")
        
        # Load pin diagram
        img = cv2.imread(pin_diagram_path)
        if img is None:
            raise ValueError(f"Could not load pin diagram from {pin_diagram_path}")
        
        # Convert to grayscale and preprocess
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Extract text from the pin diagram
        pin_text = pytesseract.image_to_string(Image.fromarray(gray))
        
        # Parse pin information
        pins = self._parse_pin_info(pin_text)
        
        # If text extraction didn't work well, try to detect pins visually
        if not pins:
            pins = self._detect_pins_visually(img, threshold)
        
        return pins
    
    def _extract_part_info(self, text):
        """Extract part information from text."""
        # Look for common sensor naming patterns
        lines = text.split('\n')
        part_info = {}
        
        # Look for part number patterns (e.g., LM35, BME280, etc.)
        part_number_pattern = r'([A-Z0-9]{2,10}[-_]?[0-9]{0,4})'
        for line in lines:
            match = re.search(part_number_pattern, line)
            if match:
                part_info["part_number"] = match.group(1)
                break
        
        return part_info
    
    def _parse_pin_info(self, text):
        """Parse pin information from extracted text."""
        lines = text.split('\n')
        pins = []
        
        # Common pin name patterns
        pin_pattern = r'(\d+)[\s.:]+([\w\+\-\/]+)'
        
        for line in lines:
            match = re.search(pin_pattern, line)
            if match:
                pin_number = match.group(1)
                pin_name = match.group(2)
                pins.append({
                    "number": pin_number,
                    "name": pin_name
                })
        
        return pins
    
    def _detect_pins_visually(self, img, threshold):
        """Use computer vision to detect pins visually."""
        pins = []
        
        # Find contours
        contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours that are likely to be pins
        pin_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 10 < area < 500:  # Adjust these thresholds based on your images
                pin_contours.append(contour)
        
        # For each potential pin, assign a number
        for i, contour in enumerate(pin_contours):
            x, y, w, h = cv2.boundingRect(contour)
            pins.append({
                "number": str(i + 1),
                "name": f"PIN{i+1}",
                "position": (x + w//2, y + h//2)
            })
        
        return pins
    
    def _get_component_dimensions(self, contour):
        """Get component dimensions from contour."""
        x, y, w, h = cv2.boundingRect(contour)
        return {"width": w, "height": h}
    
    def _determine_component_shape(self, contour):
        """Determine the component shape based on the contour."""
        num_vertices = len(contour)
        
        if num_vertices == 4:
            return "rectangular"
        elif 4 < num_vertices < 10:
            return "polygon"
        else:
            return "irregular"
    
    def generate_schematic(self, component_info, pins):
        """Generate KiCad schematic symbol (.lib file)."""
        print("Generating schematic symbol...")
        
        part_name = component_info.get("part_info", {}).get("part_number", "SENSOR")
        
        # Create schematic library content
        lib_content = f"""EESchema-LIBRARY Version 2.4
#encoding utf-8
#
# {part_name}
#
DEF {part_name} U 0 40 Y Y 1 F N
F0 "U" -200 {len(pins) * 50 + 50} 50 H V C CNN
F1 "{part_name}" 0 0 50 H V C CNN
F2 "" 0 0 50 H I C CNN
F3 "" 0 0 50 H I C CNN
DRAW
S -300 {len(pins) * 50} 300 -{len(pins) * 50} 0 1 0 f
"""
        
        # Add pins to the schematic
        for i, pin in enumerate(pins):
            pin_number = pin.get("number", str(i+1))
            pin_name = pin.get("name", f"PIN{i+1}")
            
            # Alternate left and right sides for pins
            if i % 2 == 0:
                x_pos = -400
                direction = "R"
            else:
                x_pos = 400
                direction = "L"
            
            y_pos = (len(pins) * 50) - (i * 100)
            
            lib_content += f"X {pin_name} {pin_number} {x_pos} {y_pos} 100 {direction} 50 50 1 1 B\n"
        
        # Close the schematic definition
        lib_content += "ENDDRAW\nENDDEF\n#\n#End Library\n"
        
        # Save to file
        lib_file_path = os.path.join(self.config["output_dir"], f"{part_name}.lib")
        with open(lib_file_path, "w") as f:
            f.write(lib_content)
        
        print(f"Schematic symbol generated at {lib_file_path}")
        return lib_file_path
    
    def generate_footprint(self, component_info, pins):
        """Generate KiCad footprint (.kicad_mod file)."""
        print("Generating footprint...")
        
        part_name = component_info.get("part_info", {}).get("part_number", "SENSOR")
        width = component_info["dimensions"]["width"]
        height = component_info["dimensions"]["height"]
        
        # Scale dimensions from pixels to mm (approximate)
        scale_factor = 0.1  # This is an approximation, adjust based on your images
        width_mm = width * scale_factor
        height_mm = height * scale_factor
        
        # Create footprint content
        footprint_content = f"""(module {part_name} (layer F.Cu) (tedit 0)
  (descr "Footprint for {part_name}")
  (tags "sensor")
  (attr smd)
"""
        
        # Add reference and value text
        footprint_content += f"""  (fp_text reference REF** (at 0 -{height_mm/2 + 2}) (layer F.SilkS)
    (effects (font (size 1 1) (thickness 0.15)))
  )
  (fp_text value {part_name} (at 0 {height_mm/2 + 2}) (layer F.Fab)
    (effects (font (size 1 1) (thickness 0.15)))
  )
"""
        
        # Add outline on F.SilkS
        footprint_content += f"""  (fp_line (start -{width_mm/2} -{height_mm/2}) (end {width_mm/2} -{height_mm/2}) (layer F.SilkS) (width 0.12))
  (fp_line (start {width_mm/2} -{height_mm/2}) (end {width_mm/2} {height_mm/2}) (layer F.SilkS) (width 0.12))
  (fp_line (start {width_mm/2} {height_mm/2}) (end -{width_mm/2} {height_mm/2}) (layer F.SilkS) (width 0.12))
  (fp_line (start -{width_mm/2} {height_mm/2}) (end -{width_mm/2} -{height_mm/2}) (layer F.SilkS) (width 0.12))
"""
        
        # Add pads for pins
        pin_spacing = min(width_mm, height_mm) / (len(pins) + 1)
        
        for i, pin in enumerate(pins):
            pin_number = pin.get("number", str(i+1))
            
            # Calculate pad position - distribute evenly around the component
            if component_info["shape"] == "rectangular":
                # For rectangular components, place pins along the perimeter
                if i < len(pins) / 4:  # Bottom edge
                    x = -width_mm/2 + (i+1) * (width_mm / (len(pins)/4 + 1))
                    y = height_mm/2
                elif i < len(pins) / 2:  # Right edge
                    x = width_mm/2
                    y = height_mm/2 - (i - len(pins)/4 + 1) * (height_mm / (len(pins)/4 + 1))
                elif i < 3 * len(pins) / 4:  # Top edge
                    x = width_mm/2 - (i - len(pins)/2 + 1) * (width_mm / (len(pins)/4 + 1))
                    y = -height_mm/2
                else:  # Left edge
                    x = -width_mm/2
                    y = -height_mm/2 + (i - 3*len(pins)/4 + 1) * (height_mm / (len(pins)/4 + 1))
            else:
                # For non-rectangular components, distribute pins in a circle
                angle = 2 * np.pi * i / len(pins)
                radius = min(width_mm, height_mm) / 2
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
            
            footprint_content += f"""  (pad {pin_number} smd rect (at {x:.3f} {y:.3f}) (size 1.5 0.5) (layers F.Cu F.Paste F.Mask))
"""
        
        # Close the footprint definition
        footprint_content += ")"
        
        # Save to file
        footprint_file_path = os.path.join(self.config["output_dir"], f"{part_name}.kicad_mod")
        with open(footprint_file_path, "w") as f:
            f.write(footprint_content)
        
        print(f"Footprint generated at {footprint_file_path}")
        return footprint_file_path
    
    def generate_3d_model(self, component_info, pins):
        """Generate KiCad 3D model (.wrl file)."""
        print("Generating 3D model...")
        
        part_name = component_info.get("part_info", {}).get("part_number", "SENSOR")
        width = component_info["dimensions"]["width"]
        height = component_info["dimensions"]["height"]
        
        # Scale dimensions from pixels to mm
        scale_factor = 0.1  # Approximation, adjust as needed
        width_mm = width * scale_factor
        height_mm = height * scale_factor
        
        # Determine component height based on shape
        if component_info["shape"] == "rectangular":
            # Typical IC height
            height_3d = 1.0  # 1mm height for ICs
        else:
            # For irregular shapes, make it a bit taller
            height_3d = 2.0
        
        # Generate VRML (WRL) file for the 3D model
        wrl_content = f"""#VRML V2.0 utf8
# KiCad 3D model for {part_name}
# Generated by KiCadComponentGenerator

DEF Transform_1 Transform {{
    translation 0 0 0
    rotation 0 0 1 0
    scale 1 1 1
    scaleOrientation 0 0 1 0
    center 0 0 0
    children [
        Shape {{
            appearance Appearance {{
                material Material {{
                    diffuseColor 0.1 0.1 0.1
                    shininess 0.1
                }}
            }}
            geometry Box {{
                size {width_mm} {height_mm} {height_3d}
            }}
        }}
    ]
}}
"""
        
        # Save to file
        model_file_path = os.path.join(self.config["output_dir"], f"{part_name}.wrl")
        with open(model_file_path, "w") as f:
            f.write(wrl_content)
        
        print(f"3D model generated at {model_file_path}")
        return model_file_path
    
    def process_sensor(self, image_path, pin_diagram_path):
        """Process sensor image and pin diagram to generate KiCad files."""
        try:
            # Process sensor image
            component_info = self.process_image(image_path)
            
            # Process pin diagram
            pins = self.process_pin_diagram(pin_diagram_path)
            
            # Generate KiCad files
            schematic_path = self.generate_schematic(component_info, pins)
            footprint_path = self.generate_footprint(component_info, pins)
            model_path = self.generate_3d_model(component_info, pins)
            
            return {
                "component_info": component_info,
                "pins": pins,
                "schematic_path": schematic_path,
                "footprint_path": footprint_path,
                "model_path": model_path
            }
            
        except Exception as e:
            print(f"Error processing sensor: {str(e)}")
            raise


def main():
    """Main function to run the KiCad component generator."""
    if len(sys.argv) < 3:
        print("Usage: python kicad_component_generator.py <sensor_image_path> <pin_diagram_path>")
        sys.exit(1)
    
    sensor_image_path = sys.argv[1]
    pin_diagram_path = sys.argv[2]
    
    try:
        generator = KiCadComponentGenerator()
        result = generator.process_sensor(sensor_image_path, pin_diagram_path)
        
        print("\nGeneration complete!")
        print(f"Component: {result['component_info'].get('part_info', {}).get('part_number', 'Unknown sensor')}")
        print(f"Shape: {result['component_info']['shape']}")
        print(f"Pins detected: {len(result['pins'])}")
        print(f"\nFiles generated:")
        print(f"Schematic: {result['schematic_path']}")
        print(f"Footprint: {result['footprint_path']}")
        print(f"3D Model: {result['model_path']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
