#!/usr/bin/env python3
"""
KiCad Symbol and Footprint Creator
A production-grade tool for creating custom KiCad symbols and footprints.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kicad_creator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("KiCadCreator")

# Constants
VERSION = "1.0.0"
KICAD_CONFIG_PATHS = {
    "linux": "~/.config/kicad",
    "darwin": "~/Library/Preferences/kicad",
    "win32": "~/AppData/Roaming/kicad"
}
DEFAULT_SYMBOL_LIB_TABLE = "sym-lib-table"
DEFAULT_FP_LIB_TABLE = "fp-lib-table"


@dataclass
class Point:
    """Represents a 2D point."""
    x: float
    y: float

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class SymbolPin:
    """Represents a pin for a KiCad symbol."""
    name: str
    number: str
    position: Point
    length: float = 2.54
    orientation: str = "left"  # left, right, up, down
    electrical_type: str = "passive"  # input, output, bidirectional, passive...
    visible: bool = True


@dataclass
class Symbol:
    """Represents a KiCad symbol."""
    name: str
    reference: str
    value: str
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    pins: List[SymbolPin] = field(default_factory=list)
    rectangle: Optional[Tuple[Point, Point]] = None  # (top_left, bottom_right)
    polygons: List[List[Point]] = field(default_factory=list)
    circles: List[Tuple[Point, float]] = field(default_factory=list)  # (center, radius)
    texts: List[Tuple[str, Point]] = field(default_factory=list)  # (text, position)
    library: str = ""


@dataclass
class FootprintPad:
    """Represents a pad in a KiCad footprint."""
    number: str
    position: Point
    size: Tuple[float, float]  # (width, height)
    shape: str = "rect"  # rect, circle, oval, etc.
    drill: Optional[float] = None  # For through-hole pads
    layer: str = "F.Cu F.Paste F.Mask"  # Default layers for SMD pad


@dataclass
class Footprint:
    """Represents a KiCad footprint."""
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    pads: List[FootprintPad] = field(default_factory=list)
    lines: List[Tuple[Point, Point]] = field(default_factory=list)  # (start, end)
    circles: List[Tuple[Point, float]] = field(default_factory=list)  # (center, radius)
    texts: List[Tuple[str, Point, str]] = field(default_factory=list)  # (text, position, layer)
    library: str = ""


class KiCadLibManager:
    """Handles KiCad library management operations."""
    
    def __init__(self):
        self.kicad_config_path = self._get_kicad_config_path()
        self.symbol_lib_table_path = os.path.join(self.kicad_config_path, DEFAULT_SYMBOL_LIB_TABLE)
        self.fp_lib_table_path = os.path.join(self.kicad_config_path, DEFAULT_FP_LIB_TABLE)
        self.symbol_libs = self._load_library_table(self.symbol_lib_table_path)
        self.fp_libs = self._load_library_table(self.fp_lib_table_path)
        
        logger.info(f"KiCad config path: {self.kicad_config_path}")
        logger.info(f"Found {len(self.symbol_libs)} symbol libraries")
        logger.info(f"Found {len(self.fp_libs)} footprint libraries")
    
    def _get_kicad_config_path(self) -> str:
        """Determine KiCad config path based on platform."""
        platform = sys.platform
        
        if platform in KICAD_CONFIG_PATHS:
            path = os.path.expanduser(KICAD_CONFIG_PATHS[platform])
            if os.path.exists(path):
                return path
        
        # Fallback: Ask user for KiCad config path
        logger.warning("Could not determine KiCad config path automatically")
        user_path = input("Please enter your KiCad configuration path: ")
        if os.path.exists(user_path):
            return user_path
        else:
            logger.error(f"Invalid path: {user_path}")
            raise FileNotFoundError("Invalid KiCad configuration path")
    
    def _load_library_table(self, path: str) -> Dict[str, Dict]:
        """Load library table and parse entries."""
        if not os.path.exists(path):
            logger.warning(f"Library table not found: {path}")
            return {}
        
        libs = {}
        with open(path, 'r') as f:
            content = f.read()
            
        # Simple parser for library table format
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('  (lib '):
                try:
                    # Extract name and uri
                    parts = line.split('(')
                    name_part = [p for p in parts if p.startswith('name ')][0]
                    uri_part = [p for p in parts if p.startswith('uri ')][0]
                    
                    name = name_part.split('name ')[1].split(')')[0].strip('"')
                    uri = uri_part.split('uri ')[1].split(')')[0].strip('"')
                    
                    libs[name] = {
                        'name': name,
                        'uri': uri
                    }
                except (IndexError, KeyError) as e:
                    logger.warning(f"Could not parse library entry: {line} - {e}")
        
        return libs
    
    def get_symbol_lib_path(self, lib_name: str) -> Optional[str]:
        """Get physical path for a symbol library."""
        if lib_name not in self.symbol_libs:
            return None
        
        uri = self.symbol_libs[lib_name]['uri']
        if uri.startswith('${KICAD6_SYMBOL_DIR}'):
            # Replace environment variable
            kicad_symbol_dir = os.path.join(self.kicad_config_path, 'symbols')
            return uri.replace('${KICAD6_SYMBOL_DIR}', kicad_symbol_dir)
        elif uri.startswith('${KIPRJMOD}'):
            # Project-specific library (requires project path)
            return None
        else:
            # Direct path
            return uri
    
    def get_footprint_lib_path(self, lib_name: str) -> Optional[str]:
        """Get physical path for a footprint library."""
        if lib_name not in self.fp_libs:
            return None
        
        uri = self.fp_libs[lib_name]['uri']
        if uri.startswith('${KICAD6_FOOTPRINT_DIR}'):
            # Replace environment variable
            kicad_fp_dir = os.path.join(self.kicad_config_path, 'footprints')
            return uri.replace('${KICAD6_FOOTPRINT_DIR}', kicad_fp_dir)
        elif uri.startswith('${KIPRJMOD}'):
            # Project-specific library (requires project path)
            return None
        else:
            # Direct path
            return uri
    
    def create_symbol_library(self, lib_name: str, path: str) -> bool:
        """Create a new symbol library."""
        if lib_name in self.symbol_libs:
            logger.warning(f"Symbol library already exists: {lib_name}")
            return False
        
        try:
            # Create library file if it doesn't exist
            lib_path = os.path.join(path, f"{lib_name}.kicad_sym")
            if not os.path.exists(os.path.dirname(lib_path)):
                os.makedirs(os.path.dirname(lib_path))
            
            if not os.path.exists(lib_path):
                with open(lib_path, 'w') as f:
                    f.write('(kicad_symbol_lib (version 20211014) (generator kicad_symbol_editor)\n)\n')
            
            # Add to library table
            self._add_to_library_table(self.symbol_lib_table_path, lib_name, lib_path, "Symbol")
            self.symbol_libs[lib_name] = {'name': lib_name, 'uri': lib_path}
            return True
        except Exception as e:
            logger.error(f"Failed to create symbol library: {e}")
            return False
    
    def create_footprint_library(self, lib_name: str, path: str) -> bool:
        """Create a new footprint library."""
        if lib_name in self.fp_libs:
            logger.warning(f"Footprint library already exists: {lib_name}")
            return False
        
        try:
            # Create library directory if it doesn't exist
            lib_path = os.path.join(path, lib_name)
            if not os.path.exists(lib_path):
                os.makedirs(lib_path)
            
            # Add to library table
            self._add_to_library_table(self.fp_lib_table_path, lib_name, lib_path, "Footprint")
            self.fp_libs[lib_name] = {'name': lib_name, 'uri': lib_path}
            return True
        except Exception as e:
            logger.error(f"Failed to create footprint library: {e}")
            return False
    
    def _add_to_library_table(self, table_path: str, lib_name: str, lib_path: str, lib_type: str) -> None:
        """Add a library entry to the library table file."""
        if not os.path.exists(table_path):
            # Create a new library table if it doesn't exist
            with open(table_path, 'w') as f:
                f.write(f'({"sym" if lib_type == "Symbol" else "fp"}-lib-table\n)\n')
        
        # Read current content
        with open(table_path, 'r') as f:
            content = f.read()
        
        # Add new library entry before closing parenthesis
        entry = f'  (lib (name "{lib_name}")(type "KiCad")(uri "{lib_path}")(options "")(descr ""))\n'
        if content.strip().endswith(')'):
            content = content[:-2] + entry + ')\n'
        else:
            content = content.strip() + entry + ')\n'
        
        # Write updated content
        with open(table_path, 'w') as f:
            f.write(content)


class SymbolWriter:
    """Handles writing KiCad symbol files."""
    
    def __init__(self, lib_manager: KiCadLibManager):
        self.lib_manager = lib_manager
    
    def write_symbol(self, symbol: Symbol) -> bool:
        """Write a symbol to its library file."""
        if not symbol.library:
            logger.error("Symbol library not specified")
            return False
        
        lib_path = self.lib_manager.get_symbol_lib_path(symbol.library)
        if not lib_path:
            logger.error(f"Symbol library not found: {symbol.library}")
            return False
        
        try:
            # Read existing library content
            with open(lib_path, 'r') as f:
                content = f.read()
            
            # Check if symbol already exists
            if f'(symbol "{symbol.name}"' in content:
                logger.warning(f"Symbol already exists in library: {symbol.name}")
                # Remove existing symbol
                start_idx = content.find(f'(symbol "{symbol.name}"')
                if start_idx != -1:
                    # Find matching closing parenthesis
                    depth = 0
                    end_idx = start_idx
                    for i in range(start_idx, len(content)):
                        if content[i] == '(':
                            depth += 1
                        elif content[i] == ')':
                            depth -= 1
                            if depth == 0:
                                end_idx = i + 1
                                break
                    
                    # Remove existing symbol
                    content = content[:start_idx] + content[end_idx:]
            
            # Generate symbol content
            symbol_content = self._generate_symbol_content(symbol)
            
            # Insert symbol before closing parenthesis
            if content.strip().endswith(')'):
                content = content[:-2] + symbol_content + ')\n'
            else:
                content = content.strip() + symbol_content + ')\n'
            
            # Write updated content
            with open(lib_path, 'w') as f:
                f.write(content)
            
            logger.info(f"Symbol written successfully: {symbol.name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to write symbol: {e}")
            return False
    
    def _generate_symbol_content(self, symbol: Symbol) -> str:
        """Generate KiCad symbol library content for a symbol."""
        content = []
        
        # Symbol header
        content.append(f'  (symbol "{symbol.name}" (pin_names (offset 1.016)) (in_bom yes) (on_board yes)')
        
        # Properties
        content.append(f'    (property "Reference" "{symbol.reference}" (id 0) (at 0 10.16 0)')
        content.append('      (effects (font (size 1.27 1.27)))')
        content.append('    )')
        
        content.append(f'    (property "Value" "{symbol.value}" (id 1) (at 0 -10.16 0)')
        content.append('      (effects (font (size 1.27 1.27)))')
        content.append('    )')
        
        content.append(f'    (property "Footprint" "" (id 2) (at 0 0 0)')
        content.append('      (effects (font (size 1.27 1.27)) hide)')
        content.append('    )')
        
        content.append(f'    (property "Datasheet" "" (id 3) (at 0 0 0)')
        content.append('      (effects (font (size 1.27 1.27)) hide)')
        content.append('    )')
        
        content.append(f'    (property "Description" "{symbol.description}" (id 4) (at 0 0 0)')
        content.append('      (effects (font (size 1.27 1.27)) hide)')
        content.append('    )')
        
        # Keywords
        if symbol.keywords:
            keywords = " ".join(symbol.keywords)
            content.append(f'    (property "Keywords" "{keywords}" (id 5) (at 0 0 0)')
            content.append('      (effects (font (size 1.27 1.27)) hide)')
            content.append('    )')
        
        # Symbol body - Rectangle
        if symbol.rectangle:
            top_left, bottom_right = symbol.rectangle
            content.append('    (symbol "{}_0_1"'.format(symbol.name))
            content.append('      (rectangle (start {} {}) (end {} {}) (stroke (width 0.1524) (type default) (color 0 0 0 0))'.format(
                top_left.x, top_left.y, bottom_right.x, bottom_right.y))
            content.append('        (fill (type background))')
            content.append('      )')
            content.append('    )')
        
        # Polygons
        if symbol.polygons:
            content.append('    (symbol "{}_0_2"'.format(symbol.name))
            for poly in symbol.polygons:
                points_str = " ".join([f"(xy {p.x} {p.y})" for p in poly])
                content.append('      (polyline')
                content.append('        (pts ' + points_str + ')')
                content.append('        (stroke (width 0.1524) (type default) (color 0 0 0 0))')
                content.append('        (fill (type none))')
                content.append('      )')
            content.append('    )')
        
        # Circles
        if symbol.circles:
            content.append('    (symbol "{}_0_3"'.format(symbol.name))
            for center, radius in symbol.circles:
                content.append('      (circle (center {} {}) (radius {}) (stroke (width 0.1524) (type default) (color 0 0 0 0))'.format(
                    center.x, center.y, radius))
                content.append('        (fill (type none))')
                content.append('      )')
            content.append('    )')
        
        # Texts
        if symbol.texts:
            content.append('    (symbol "{}_0_4"'.format(symbol.name))
            for text, position in symbol.texts:
                content.append('      (text "{}" (at {} {}) (effects (font (size 1.27 1.27)))'.format(
                    text, position.x, position.y))
                content.append('      )')
            content.append('    )')
        
        # Pins
        if symbol.pins:
            content.append('    (symbol "{}_1_1"'.format(symbol.name))
            for pin in symbol.pins:
                visible = "visible" if pin.visible else "hide"
                content.append('      (pin {} "{}" (at {} {} 0) (length {})'.format(
                    pin.electrical_type, pin.name, pin.position.x, pin.position.y, pin.length))
                content.append('        (name (effects (font (size 1.27 1.27)) {}))'.format(visible))
                content.append('        (number (effects (font (size 1.27 1.27)) {}))'.format(visible))
                content.append('      )')
            content.append('    )')
        
        # Close symbol
        content.append('  )')
        
        return '\n'.join(content)


class FootprintWriter:
    """Handles writing KiCad footprint files."""
    
    def __init__(self, lib_manager: KiCadLibManager):
        self.lib_manager = lib_manager
    
    def write_footprint(self, footprint: Footprint) -> bool:
        """Write a footprint to its library directory."""
        if not footprint.library:
            logger.error("Footprint library not specified")
            return False
        
        lib_path = self.lib_manager.get_footprint_lib_path(footprint.library)
        if not lib_path:
            logger.error(f"Footprint library not found: {footprint.library}")
            return False
        
        try:
            # Create footprint file path
            fp_file = os.path.join(lib_path, f"{footprint.name}.kicad_mod")
            
            # Generate footprint content
            fp_content = self._generate_footprint_content(footprint)
            
            # Write footprint file
            with open(fp_file, 'w') as f:
                f.write(fp_content)
            
            logger.info(f"Footprint written successfully: {footprint.name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to write footprint: {e}")
            return False
    
    def _generate_footprint_content(self, fp: Footprint) -> str:
        """Generate KiCad footprint content."""
        content = []
        
        # Footprint header
        content.append('(footprint "{}" (layer "F.Cu")'.format(fp.name))
        content.append('  (tedit {})'.format(hex(int(10000))[2:]))  # Placeholder edit time
        
        # Properties
        content.append('  (descr "{}")'.format(fp.description))
        
        if fp.tags:
            content.append('  (tags "{}")'.format(" ".join(fp.tags)))
        
        # Default text items
        content.append('  (attr smd)')
        content.append('  (fp_text reference "REF**" (at 0 -3 0) (layer "F.SilkS")')
        content.append('    (effects (font (size 1 1) (thickness 0.15)))')
        content.append('  )')
        
        content.append('  (fp_text value "{}" (at 0 3 0) (layer "F.Fab")'.format(fp.name))
        content.append('    (effects (font (size 1 1) (thickness 0.15)))')
        content.append('  )')
        
        # Lines
        for start, end in fp.lines:
            content.append('  (fp_line (start {} {}) (end {} {}) (layer "F.SilkS") (width 0.12))'.format(
                start.x, start.y, end.x, end.y))
        
        # Circles
        for center, radius in fp.circles:
            content.append('  (fp_circle (center {} {}) (end {} {}) (layer "F.SilkS") (width 0.12))'.format(
                center.x, center.y, center.x + radius, center.y))
        
        # Custom text elements
        for text, position, layer in fp.texts:
            content.append('  (fp_text user "{}" (at {} {} 0) (layer "{}")'.format(
                text, position.x, position.y, layer))
            content.append('    (effects (font (size 1 1) (thickness 0.15)))')
            content.append('  )')
        
        # Pads
        for pad in fp.pads:
            if pad.drill:  # Through-hole pad
                content.append('  (pad "{}" thru_hole {} (at {} {}) (size {} {}) (drill {}) (layers *.Cu *.Mask))'.format(
                    pad.number, pad.shape, pad.position.x, pad.position.y, 
                    pad.size[0], pad.size[1], pad.drill))
            else:  # SMD pad
                content.append('  (pad "{}" smd {} (at {} {}) (size {} {}) (layers {}))'.format(
                    pad.number, pad.shape, pad.position.x, pad.position.y, 
                    pad.size[0], pad.size[1], pad.layer))
        
        # Close footprint
        content.append(')')
        
        return '\n'.join(content)


class SymbolGenerator:
    """Utility class to generate common symbol types."""
    
    @staticmethod
    def create_ic(name: str, pins: List[Dict], width: float = 10.16, 
                  reference: str = "U", description: str = "", keywords: List[str] = None) -> Symbol:
        """Create a rectangular IC symbol with pins."""
        if keywords is None:
            keywords = []
        
        # Calculate height based on number of pins
        left_pins = [p for p in pins if p.get("side", "left") == "left"]
        right_pins = [p for p in pins if p.get("side", "right") == "right"]
        top_pins = [p for p in pins if p.get("side", "top") == "top"]
        bottom_pins = [p for p in pins if p.get("side", "bottom") == "bottom"]
        
        max_pins_per_side = max(len(left_pins), len(right_pins))
        height = max(max_pins_per_side * 2.54, 10.16)
        
        # Create symbol
        symbol = Symbol(
            name=name,
            reference=reference,
            value=name,
            description=description,
            keywords=keywords,
            rectangle=(Point(-width/2, height/2), Point(width/2, -height/2))
        )
        
        # Add pins
        pin_spacing = 2.54
        
        # Left side pins
        for i, pin_def in enumerate(left_pins):
            y_pos = height/2 - pin_spacing * (i + 0.5)
            pin = SymbolPin(
                name=pin_def["name"],
                number=pin_def["number"],
                position=Point(-width/2 - pin_spacing, y_pos),
                orientation="right",
                electrical_type=pin_def.get("type", "passive")
            )
            symbol.pins.append(pin)
        
        # Right side pins
        for i, pin_def in enumerate(right_pins):
            y_pos = height/2 - pin_spacing * (i + 0.5)
            pin = SymbolPin(
                name=pin_def["name"],
                number=pin_def["number"],
                position=Point(width/2 + pin_spacing, y_pos),
                orientation="left",
                electrical_type=pin_def.get("type", "passive")
            )
            symbol.pins.append(pin)
        
        # Top pins
        if top_pins:
            pin_width = width / (len(top_pins) + 1)
            for i, pin_def in enumerate(top_pins):
                x_pos = -width/2 + pin_width * (i + 1)
                pin = SymbolPin(
                    name=pin_def["name"],
                    number=pin_def["number"],
                    position=Point(x_pos, height/2 + pin_spacing),
                    orientation="down",
                    electrical_type=pin_def.get("type", "passive")
                )
                symbol.pins.append(pin)
        
        # Bottom pins
        if bottom_pins:
            pin_width = width / (len(bottom_pins) + 1)
            for i, pin_def in enumerate(bottom_pins):
                x_pos = -width/2 + pin_width * (i + 1)
                pin = SymbolPin(
                    name=pin_def["name"],
                    number=pin_def["number"],
                    position=Point(x_pos, -height/2 - pin_spacing),
                    orientation="up",
                    electrical_type=pin_def.get("type", "passive")
                )
                symbol.pins.append(pin)
        
        return symbol
    
    @staticmethod
    def create_resistor(name: str, reference: str = "R", value: str = "R", 
                       description: str = "Resistor", keywords: List[str] = None) -> Symbol:
        """Create a resistor symbol."""
        if keywords is None:
            keywords = ["resistor", "passive"]
        
        symbol = Symbol(
            name=name,
            reference=reference,
            value=value,
            description=description,
            keywords=keywords
        )
        
        # Add pins
        symbol.pins = [
            SymbolPin(name="1", number="1", position=Point(-5.08, 0), orientation="right"),
            SymbolPin(name="2", number="2", position=Point(5.08, 0), orientation="left")
        ]
        
        # Add rectangle
        symbol.rectangle = (Point(-2.54, 1.27), Point(2.54, -1.27))
        
        return symbol
    
    @staticmethod
    def create_capacitor(name: str, polarized: bool = False, reference: str = "C", 
                        value: str = "C", description: str = "Capacitor", 
                        keywords: List[str] = None) -> Symbol:
        """Create a capacitor symbol."""
        if keywords is None:
            keywords = ["capacitor", "passive"]
            if polarized:
                keywords.append("polarized")
        
        symbol = Symbol(
            name=name,
            reference=reference,
            value=value,
            description=description,
            keywords=keywords
        )
        
        # Add pins
        symbol.pins = [
            SymbolPin(name="1", number="1", position=Point(-2.54, 0), orientation="right"),
            SymbolPin(name="2", number="2", position=Point(2.54, 0), orientation="left")
        ]
        
        # For non-polarized capacitor - two parallel lines
        if not polarized:
            symbol.polygons = [
                [Point(-0.508, 1.27), Point(-0.508, -1.27)],
                [Point(0.508, 1.27), Point(0.508, -1.27)]
            ]
        else:
            # For polarized capacitor - one curved line and "+" symbol
            symbol.polygons = [
                [Point(-0.508, 1.27), Point(-0.508, -1.27)],
            ]
            # Curved line
            symbol.polygons.append([
                Point(0.508, 1.27), Point(0.508, -1.27)
            ])
            # "+" symbol
            symbol.texts.append(("+", Point(-1.27, 1.27)))
        
        return symbol
    
    @staticmethod
    def create_connector(name: str, pins: int, reference: str = "J", dual_row: bool = False,
                        description: str = "", keywords: List[str] = None) -> Symbol:
        """Create a connector symbol with specified number of pins."""
        if keywords is None:
            keywords = ["connector"]
        
        rows = 2 if dual_row else 1
        pins_per_row = pins // rows if dual_row else pins
        height = pins_per_row * 2.54
        
        symbol = Symbol(
            name=name,
            reference=reference,
            value=name,
            description=description or f"{pins}-pin connector",
            keywords=keywords,
            rectangle=(Point(-5.08, height/2), Point(5.08, -height/2 + (0 if pins_per_row % 2 == 0 else 1.27)))
        )
        
        # Add pins
        for i in range(pins):
            if dual_row:
                # Dual row: even pins on right, odd pins on left
                if i % 2 == 0:  # Left side (odd pins in 1-based indexing)
                    position = Point(-7.62, height/2 - (i // 2) * 2.54)
                    orientation = "right"
                else:  # Right side (even pins in 1-based indexing)
                    position = Point(7.62, height/2 - (i // 2) * 2.54)
                    orientation = "left"
            else:
                # Single row: all pins on left side
                position = Point(-7.62, height/2 - i * 2.54)
                orientation = "right"
            
            pin = SymbolPin(
                name=str(i+1),
                number=str(i+1),
                position=position,
                orientation=orientation,
                electrical_type="passive"
            )
            symbol.pins.append(pin)
        
        return symbol


class FootprintGenerator:
    """Utility class to generate common footprint types."""
    
    @staticmethod
    def create_smd_resistor(name: str, size: str = "0805", 
                          description: str = "SMD Resistor", tags: List[str] = None) -> Footprint:
        """Create an SMD resistor footprint."""
        if tags is None:
            tags = ["resistor", "smd"]
        
        # Common SMD sizes in mm (length, width, pad width, pad height)
        smd_sizes = {
            "0402": (1.0, 0.5, 0.5, 0.5),
            "0603": (1.6, 0.8, 0.8, 0.8),
            "0805": (2.0, 1.25, 1.0, 1.3),
            "1206": (3.2, 1.6, 1.6, 1.6),
            "1210": (3.2, 2.5, 1.6, 2.5),
            "2010": (5.0, 2.5, 2.0, 2.5),
            "2512": (6.3, 3.2, 2.2, 3.2)
        }
        
        if size not in smd_sizes:
            size = "0805"  # Default
        
        length, width, pad_width, pad_height = smd_sizes[size]
        pad_distance = length * 0.8  # Distance between pad centers
        
        footprint = Footprint(
            name=f"{name}_{size}",
            description=f"{description}, {size} package",
            tags=tags + [size]
        )
        
        # Add pads
        footprint.pads = [
            FootprintPad(
                number="1",
                position=Point(-pad_distance/2, 0),
                size=(pad_width, pad_height),
                shape="rect"
            ),
            FootprintPad(
                number="2",
                position=Point(pad_distance/2, 0),
                size=(pad_width, pad_height),
                shape="rect"
            )
        ]
        
        # Add body outline
        footprint.lines = [
            (Point(-length/2, width/2), Point(length/2, width/2)),
            (Point(length/2, width/2), Point(length/2, -width/2)),
            (Point(length/2, -width/2), Point(-length/2, -width/2)),
            (Point(-length/2, -width/2), Point(-length/2, width/2))
        ]
        
        return footprint
    
    @staticmethod
    def create_smd_capacitor(name: str, size: str = "0805", polarized: bool = False,
                           description: str = "SMD Capacitor", tags: List[str] = None) -> Footprint:
        """Create an SMD capacitor footprint."""
        if tags is None:
            tags = ["capacitor", "smd"]
            if polarized:
                tags.append("polarized")
        
        # Reuse resistor footprint dimensions
        fp = FootprintGenerator.create_smd_resistor(
            name=name,
            size=size,
            description=description,
            tags=tags
        )
        
        # For polarized, add polarity marking
        if polarized:
            pad_distance = float(fp.pads[1].position.x - fp.pads[0].position.x)
            width = fp.pads[0].size[1]
            
            # Add a plus symbol near the positive pad
            fp.texts.append((
                "+",
                Point(-pad_distance/2, -width - 0.5),
                "F.SilkS"
            ))
        
        return fp
    
    @staticmethod
    def create_sot23(name: str, pins: int = 3, description: str = "SOT-23 package", 
                   tags: List[str] = None) -> Footprint:
        """Create an SOT-23 footprint."""
        if tags is None:
            tags = ["SOT-23", "SOT23"]
        
        # SOT-23 dimensions (width, height, pad width, pad height, pad distance)
        width = 2.9
        height = 3.0 if pins > 3 else 1.6
        pad_width = 0.8
        pad_height = 0.9
        pad_distance_x = 1.9
        pad_distance_y = 2.6 if pins > 3 else 0.95
        
        footprint = Footprint(
            name=name,
            description=description,
            tags=tags
        )
        
        # Add pads
        if pins == 3:
            # Standard SOT-23 with 3 pins
            footprint.pads = [
                FootprintPad(
                    number="1",
                    position=Point(-pad_distance_x/2, pad_distance_y),
                    size=(pad_width, pad_height),
                    shape="rect"
                ),
                FootprintPad(
                    number="2",
                    position=Point(pad_distance_x/2, pad_distance_y),
                    size=(pad_width, pad_height),
                    shape="rect"
                ),
                FootprintPad(
                    number="3",
                    position=Point(0, -pad_distance_y),
                    size=(pad_width, pad_height),
                    shape="rect"
                )
            ]
        elif pins == 5:
            # SOT-23-5
            footprint.pads = [
                FootprintPad(
                    number="1",
                    position=Point(-1.3, pad_distance_y),
                    size=(pad_width, pad_height),
                    shape="rect"
                ),
                FootprintPad(
                    number="2",
                    position=Point(0, pad_distance_y),
                    size=(pad_width, pad_height),
                    shape="rect"
                ),
                FootprintPad(
                    number="3",
                    position=Point(1.3, pad_distance_y),
                    size=(pad_width, pad_height),
                    shape="rect"
                ),
                FootprintPad(
                    number="4",
                    position=Point(1.3, -pad_distance_y),
                    size=(pad_width, pad_height),
                    shape="rect"
                ),
                FootprintPad(
                    number="5",
                    position=Point(-1.3, -pad_distance_y),
                    size=(pad_width, pad_height),
                    shape="rect"
                )
            ]
        elif pins == 6:
            # SOT-23-6
            footprint.pads = [
                FootprintPad(
                    number="1",
                    position=Point(-1.3, pad_distance_y),
                    size=(pad_width, pad_height),
                    shape="rect"
                ),
                FootprintPad(
                    number="2",
                    position=Point(0, pad_distance_y),
                    size=(pad_width, pad_height),
                    shape="rect"
                ),
                FootprintPad(
                    number="3",
                    position=Point(1.3, pad_distance_y),
                    size=(pad_width, pad_height),
                    shape="rect"
                ),
                FootprintPad(
                    number="4",
                    position=Point(1.3, -pad_distance_y),
                    size=(pad_width, pad_height),
                    shape="rect"
                ),
                FootprintPad(
                    number="5",
                    position=Point(0, -pad_distance_y),
                    size=(pad_width, pad_height),
                    shape="rect"
                ),
                FootprintPad(
                    number="6",
                    position=Point(-1.3, -pad_distance_y),
                    size=(pad_width, pad_height),
                    shape="rect"
                )
            ]
        
        # Add body outline
        footprint.lines = [
            (Point(-width/2, height/2), Point(width/2, height/2)),
            (Point(width/2, height/2), Point(width/2, -height/2)),
            (Point(width/2, -height/2), Point(-width/2, -height/2)),
            (Point(-width/2, -height/2), Point(-width/2, height/2))
        ]
        
        # Add pin 1 marker
        footprint.circles.append((Point(-width/2 - 0.3, pad_distance_y), 0.1))
        
        return footprint
    
    @staticmethod
    def create_dip(name: str, pins: int = 8, pitch: float = 2.54, width: float = 7.62,
                 description: str = "DIP package", tags: List[str] = None) -> Footprint:
        """Create a DIP footprint."""
        if tags is None:
            tags = ["DIP", "DIL", "PDIP"]
        
        # DIP dimensions
        rows = 2
        pins_per_row = pins // 2
        length = (pins_per_row - 1) * pitch
        drill = 0.8
        pad_size = (1.6, 1.6)
        
        footprint = Footprint(
            name=name,
            description=description,
            tags=tags
        )
        
        # Add pads
        for i in range(pins):
            if i < pins_per_row:
                # Left row
                position = Point(-width/2, length/2 - i * pitch)
                number = str(i + 1)
            else:
                # Right row
                position = Point(width/2, -length/2 + (i - pins_per_row) * pitch)
                number = str(pins - (i - pins_per_row))
            
            footprint.pads.append(
                FootprintPad(
                    number=number,
                    position=position,
                    size=pad_size,
                    shape="circle",
                    drill=drill
                )
            )
        
        # Add body outline
        body_width = width - 1.0
        body_length = length + 1.0
        
        footprint.lines = [
            (Point(-body_width/2, body_length/2), Point(body_width/2, body_length/2)),
            (Point(body_width/2, body_length/2), Point(body_width/2, -body_length/2)),
            (Point(body_width/2, -body_length/2), Point(-body_width/2, -body_length/2)),
            (Point(-body_width/2, -body_length/2), Point(-body_width/2, body_length/2))
        ]
        
        # Add notch for pin 1
        notch_width = body_width * 0.2
        notch_depth = 1.0
        
        footprint.lines.extend([
            (Point(-notch_width, body_length/2), Point(-notch_width, body_length/2 + notch_depth)),
            (Point(-notch_width, body_length/2 + notch_depth), Point(notch_width, body_length/2 + notch_depth)),
            (Point(notch_width, body_length/2 + notch_depth), Point(notch_width, body_length/2))
        ])
        
        return footprint
    
    @staticmethod
    def create_connector_header(name: str, pins: int, rows: int = 1, pitch: float = 2.54,
                              description: str = "", tags: List[str] = None) -> Footprint:
        """Create a connector header footprint."""
        if tags is None:
            tags = ["connector", "header", "pin header"]
        
        if rows < 1:
            rows = 1
        
        pins_per_row = pins // rows
        length = (pins_per_row - 1) * pitch
        width = (rows - 1) * pitch
        drill = 1.0
        pad_size = (1.8, 1.8)
        
        footprint = Footprint(
            name=name,
            description=description or f"{pins}-pin connector header, {rows}x{pins_per_row}",
            tags=tags
        )
        
        # Add pads
        pin_number = 1
        for row in range(rows):
            for col in range(pins_per_row):
                position = Point(row * pitch, length/2 - col * pitch)
                
                footprint.pads.append(
                    FootprintPad(
                        number=str(pin_number),
                        position=position,
                        size=pad_size,
                        shape="circle",
                        drill=drill
                    )
                )
                pin_number += 1
        
        # Add outline
        outline_width = width + 1.5
        outline_length = length + 1.5
        
        footprint.lines = [
            (Point(-1.5/2, outline_length/2), Point(outline_width, outline_length/2)),
            (Point(outline_width, outline_length/2), Point(outline_width, -outline_length/2)),
            (Point(outline_width, -outline_length/2), Point(-1.5/2, -outline_length/2)),
            (Point(-1.5/2, -outline_length/2), Point(-1.5/2, outline_length/2))
        ]
        
        # Add pin 1 marker
        footprint.circles.append((Point(-1.0, length/2 + 1.0), 0.3))
        
        return footprint


class GUI(tk.Tk):
    """GUI for the KiCad Symbol and Footprint Creator."""
    
    def __init__(self):
        super().__init__()
        
        self.title(f"KiCad Symbol and Footprint Creator v{VERSION}")
        self.geometry("900x700")
        
        # Initialize library manager
        try:
            self.lib_manager = KiCadLibManager()
            self.symbol_writer = SymbolWriter(self.lib_manager)
            self.footprint_writer = FootprintWriter(self.lib_manager)
            
            self._create_widgets()
            self._setup_layout()
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize: {e}")
            self.destroy()
    
    def _create_widgets(self):
        """Create GUI widgets."""
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self)
        
        # Symbol tab
        self.symbol_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.symbol_frame, text="Symbols")
        
        # Footprint tab
        self.footprint_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.footprint_frame, text="Footprints")
        
        # Library management tab
        self.library_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.library_frame, text="Libraries")
        
        # About tab
        self.about_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.about_frame, text="About")
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        
        # Create widget content for each tab
        self._create_symbol_tab()
        self._create_footprint_tab()
        self._create_library_tab()
        self._create_about_tab()
    
    def _setup_layout(self):
        """Set up widget layout."""
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=2)
    
    def _create_symbol_tab(self):
        """Create the symbol tab content."""
        # Symbol types frame
        symbol_types_frame = ttk.LabelFrame(self.symbol_frame, text="Symbol Type")
        symbol_types_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.symbol_type_var = tk.StringVar(value="ic")
        symbol_types = [
            ("IC / Microcontroller", "ic"),
            ("Resistor", "resistor"),
            ("Capacitor", "capacitor"),
            ("Connector", "connector")
        ]
        
        # Radio buttons for symbol types
        for text, value in symbol_types:
            rb = ttk.Radiobutton(symbol_types_frame, text=text, value=value, variable=self.symbol_type_var)
            rb.pack(anchor=tk.W, padx=10, pady=2)
            rb.bind("<ButtonRelease-1>", self._on_symbol_type_change)
        
        # Symbol details frame
        self.symbol_details_frame = ttk.LabelFrame(self.symbol_frame, text="Symbol Details")
        self.symbol_details_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Common fields for all symbol types
        ttk.Label(self.symbol_details_frame, text="Symbol Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.symbol_name_var = tk.StringVar()
        ttk.Entry(self.symbol_details_frame, textvariable=self.symbol_name_var).grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        
        ttk.Label(self.symbol_details_frame, text="Library:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.symbol_lib_var = tk.StringVar()
        self.symbol_lib_combo = ttk.Combobox(self.symbol_details_frame, textvariable=self.symbol_lib_var)
        self.symbol_lib_combo["values"] = list(self.lib_manager.symbol_libs.keys())
        self.symbol_lib_combo.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        
        ttk.Label(self.symbol_details_frame, text="Reference:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.symbol_ref_var = tk.StringVar(value="U")
        ttk.Entry(self.symbol_details_frame, textvariable=self.symbol_ref_var).grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        
        ttk.Label(self.symbol_details_frame, text="Description:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.symbol_desc_var = tk.StringVar()
        ttk.Entry(self.symbol_details_frame, textvariable=self.symbol_desc_var).grid(row=3, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        
        ttk.Label(self.symbol_details_frame, text="Keywords:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.symbol_keywords_var = tk.StringVar()
        ttk.Entry(self.symbol_details_frame, textvariable=self.symbol_keywords_var).grid(row=4, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        
        # Frame for type-specific options
        self.symbol_options_frame = ttk.Frame(self.symbol_details_frame)
        self.symbol_options_frame.grid(row=5, column=0, columnspan=2, sticky=tk.W+tk.E+tk.N+tk.S, padx=5, pady=5)
        
        # Initial options display
        self._show_ic_options()
        
        # Buttons
        button_frame = ttk.Frame(self.symbol_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Create Symbol", command=self._create_symbol).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Preview").pack(side=tk.RIGHT, padx=5)
    
    def _create_footprint_tab(self):
        """Create the footprint tab content."""
        # Footprint types frame
        fp_types_frame = ttk.LabelFrame(self.footprint_frame, text="Footprint Type")
        fp_types_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.fp_type_var = tk.StringVar(value="smd_resistor")
        fp_types = [
            ("SMD Resistor", "smd_resistor"),
            ("SMD Capacitor", "smd_capacitor"),
            ("SOT-23 Package", "sot23"),
            ("DIP Package", "dip"),
            ("Header Connector", "header")
        ]
        
        # Radio buttons for footprint types
        for text, value in fp_types:
            rb = ttk.Radiobutton(fp_types_frame, text=text, value=value, variable=self.fp_type_var)
            rb.pack(anchor=tk.W, padx=10, pady=2)
            rb.bind("<ButtonRelease-1>", self._on_footprint_type_change)
        
        # Footprint details frame
        self.fp_details_frame = ttk.LabelFrame(self.footprint_frame, text="Footprint Details")
        self.fp_details_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Common fields for all footprint types
        ttk.Label(self.fp_details_frame, text="Footprint Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.fp_name_var = tk.StringVar()
        ttk.Entry(self.fp_details_frame, textvariable=self.fp_name_var).grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        
        ttk.Label(self.fp_details_frame, text="Library:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.fp_lib_var = tk.StringVar()
        self.fp_lib_combo = ttk.Combobox(self.fp_details_frame, textvariable=self.fp_lib_var)
        self.fp_lib_combo["values"] = list(self.lib_manager.fp_libs.keys())
        self.fp_lib_combo.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        
        ttk.Label(self.fp_details_frame, text="Description:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.fp_desc_var = tk.StringVar()
        ttk.Entry(self.fp_details_frame, textvariable=self.fp_desc_var).grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        
        ttk.Label(self.fp_details_frame, text="Tags:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.fp_tags_var = tk.StringVar()
        ttk.Entry(self.fp_details_frame, textvariable=self.fp_tags_var).grid(row=3, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        
        # Frame for type-specific options
        self.fp_options_frame = ttk.Frame(self.fp_details_frame)
        self.fp_options_frame.grid(row=4, column=0, columnspan=2, sticky=tk.W+tk.E+tk.N+tk.S, padx=5, pady=5)
        
        # Initial options display
        self._show_smd_resistor_options()
        
        # Buttons
        button_frame = ttk.Frame(self.footprint_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Create Footprint", command=self._create_footprint).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Preview").pack(side=tk.RIGHT, padx=5)
    
    def _create_library_tab(self):
        """Create the library management tab content."""
        # Symbol libraries frame
        sym_lib_frame = ttk.LabelFrame(self.library_frame, text="Symbol Libraries")
        sym_lib_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Symbol libraries listbox
        self.sym_lib_listbox = tk.Listbox(sym_lib_frame)
        self.sym_lib_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._populate_sym_lib_list()
        
        # Symbol library buttons
        sym_lib_button_frame = ttk.Frame(sym_lib_frame)
        sym_lib_button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        ttk.Button(sym_lib_button_frame, text="Add Library", command=self._add_symbol_library).pack(fill=tk.X, pady=2)
        ttk.Button(sym_lib_button_frame, text="Open in KiCad").pack(fill=tk.X, pady=2)
        
        # Footprint libraries frame
        fp_lib_frame = ttk.LabelFrame(self.library_frame, text="Footprint Libraries")
        fp_lib_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Footprint libraries listbox
        self.fp_lib_listbox = tk.Listbox(fp_lib_frame)
        self.fp_lib_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._populate_fp_lib_list()
        
        # Footprint library buttons
        fp_lib_button_frame = ttk.Frame(fp_lib_frame)
        fp_lib_button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        ttk.Button(fp_lib_button_frame, text="Add Library", command=self._add_footprint_library).pack(fill=tk.X, pady=2)
        ttk.Button(fp_lib_button_frame, text="Open in KiCad").pack(fill=tk.X, pady=2)
    
    def _create_about_tab(self):
        """Create the about tab content."""
        about_text = f"""
        KiCad Symbol and Footprint Creator v{VERSION}
        
        A tool for creating custom KiCad symbols and footprints.
        
        Features:
        - Create standard symbol types (ICs, resistors, capacitors, connectors)
        - Create standard footprint types (SMD, DIP, SOT, headers)
        - Manage KiCad libraries
        
        This tool integrates with KiCad's library structure and provides
        a user-friendly interface for creating custom components.
        """
        
        about_label = ttk.Label(self.about_frame, text=about_text, justify=tk.LEFT, wraplength=500)
        about_label.pack(padx=20, pady=20)
    
    def _on_symbol_type_change(self, event):
        """Handle symbol type change."""
        # Wait for the radiobutton to be released
        self.after(100, self._update_symbol_options)
    
    def _update_symbol_options(self):
        """Update the symbol options based on the selected type."""
        symbol_type = self.symbol_type_var.get()
        
        # Clear the options frame
        for widget in self.symbol_options_frame.winfo_children():
            widget.destroy()
        
        # Set default reference based on symbol type
        if symbol_type == "resistor":
            self.symbol_ref_var.set("R")
        elif symbol_type == "capacitor":
            self.symbol_ref_var.set("C")
        elif symbol_type == "connector":
            self.symbol_ref_var.set("J")
        else:
            self.symbol_ref_var.set("U")
        
        # Show options based on symbol type
        if symbol_type == "ic":
            self._show_ic_options()
        elif symbol_type == "resistor":
            self._show_resistor_options()
        elif symbol_type == "capacitor":
            self._show_capacitor_options()
        elif symbol_type == "connector":
            self._show_connector_options()
    
    def _show_ic_options(self):
        """Show options for IC symbol type."""
        # Pin definition frame
        pin_frame = ttk.LabelFrame(self.symbol_options_frame, text="Pins")
        pin_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Number of pins
        ttk.Label(pin_frame, text="Number of Left Pins:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.left_pins_var = tk.IntVar(value=8)
        ttk.Spinbox(pin_frame, from_=0, to=50, textvariable=self.left_pins_var, width=5).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(pin_frame, text="Number of Right Pins:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.right_pins_var = tk.IntVar(value=8)
        ttk.Spinbox(pin_frame, from_=0, to=50, textvariable=self.right_pins_var, width=5).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(pin_frame, text="Number of Top Pins:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.top_pins_var = tk.IntVar(value=0)
        ttk.Spinbox(pin_frame, from_=0, to=50, textvariable=self.top_pins_var, width=5).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(pin_frame, text="Number of Bottom Pins:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.bottom_pins_var = tk.IntVar(value=0)
        ttk.Spinbox(pin_frame, from_=0, to=50, textvariable=self.bottom_pins_var, width=5).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Symbol dimensions
        ttk.Label(pin_frame, text="Width (mm):").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.ic_width_var = tk.DoubleVar(value=20.0)
        ttk.Spinbox(pin_frame, from_=5.0, to=100.0, increment=0.5, textvariable=self.ic_width_var, width=5).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(pin_frame, text="Height (mm):").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        self.ic_height_var = tk.DoubleVar(value=20.0)
        ttk.Spinbox(pin_frame, from_=5.0, to=100.0, increment=0.5, textvariable=self.ic_height_var, width=5).grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Pin naming
        ttk.Label(pin_frame, text="Pin Naming:").grid(row=2, column=2, sticky=tk.W, padx=5, pady=2)
        self.pin_naming_var = tk.StringVar(value="numeric")
        naming_combo = ttk.Combobox(pin_frame, textvariable=self.pin_naming_var, width=10)
        naming_combo["values"] = ["numeric", "letters", "custom"]
        naming_combo.grid(row=2, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Pin spacing
        ttk.Label(pin_frame, text="Pin Spacing:").grid(row=3, column=2, sticky=tk.W, padx=5, pady=2)
        self.pin_spacing_var = tk.DoubleVar(value=2.54)
        ttk.Spinbox(pin_frame, from_=1.0, to=10.0, increment=0.1, textvariable=self.pin_spacing_var, width=5).grid(row=3, column=3, sticky=tk.W, padx=5, pady=2)
    
    def _show_resistor_options(self):
        """Show options for resistor symbol type."""
        # Resistor options frame
        res_frame = ttk.LabelFrame(self.symbol_options_frame, text="Resistor Options")
        res_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Style
        ttk.Label(res_frame, text="Style:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.res_style_var = tk.StringVar(value="IEC")
        style_combo = ttk.Combobox(res_frame, textvariable=self.res_style_var, width=15)
        style_combo["values"] = ["IEC", "US", "VARIABLE"]
        style_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Power rating
        ttk.Label(res_frame, text="Power Rating:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.res_power_var = tk.StringVar()
        ttk.Entry(res_frame, textvariable=self.res_power_var, width=15).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Tolerance
        ttk.Label(res_frame, text="Tolerance (%):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.res_tolerance_var = tk.StringVar()
        ttk.Entry(res_frame, textvariable=self.res_tolerance_var, width=15).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
    
    def _show_capacitor_options(self):
        """Show options for capacitor symbol type."""
        # Capacitor options frame
        cap_frame = ttk.LabelFrame(self.symbol_options_frame, text="Capacitor Options")
        cap_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Style
        ttk.Label(cap_frame, text="Style:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.cap_style_var = tk.StringVar(value="STANDARD")
        style_combo = ttk.Combobox(cap_frame, textvariable=self.cap_style_var, width=15)
        style_combo["values"] = ["STANDARD", "POLARIZED", "VARIABLE"]
        style_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Voltage rating
        ttk.Label(cap_frame, text="Voltage Rating:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.cap_voltage_var = tk.StringVar()
        ttk.Entry(cap_frame, textvariable=self.cap_voltage_var, width=15).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Tolerance
        ttk.Label(cap_frame, text="Tolerance (%):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.cap_tolerance_var = tk.StringVar()
        ttk.Entry(cap_frame, textvariable=self.cap_tolerance_var, width=15).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
    
    def _show_connector_options(self):
        """Show options for connector symbol type."""
        # Connector options frame
        conn_frame = ttk.LabelFrame(self.symbol_options_frame, text="Connector Options")
        conn_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Number of pins
        ttk.Label(conn_frame, text="Number of Pins:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.conn_pins_var = tk.IntVar(value=8)
        ttk.Spinbox(conn_frame, from_=1, to=100, textvariable=self.conn_pins_var, width=5).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Style
        ttk.Label(conn_frame, text="Style:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.conn_style_var = tk.StringVar(value="SINGLE_ROW")
        style_combo = ttk.Combobox(conn_frame, textvariable=self.conn_style_var, width=15)
        style_combo["values"] = ["SINGLE_ROW", "DOUBLE_ROW", "GRID"]
        style_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Pin layout
        ttk.Label(conn_frame, text="Pin Layout:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.conn_layout_var = tk.StringVar(value="LEFT")
        layout_combo = ttk.Combobox(conn_frame, textvariable=self.conn_layout_var, width=15)
        layout_combo["values"] = ["LEFT", "RIGHT", "TOP", "BOTTOM"]
        layout_combo.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
    
    def _on_footprint_type_change(self, event):
        """Handle footprint type change."""
        # Wait for the radiobutton to be released
        self.after(100, self._update_footprint_options)
    
    def _update_footprint_options(self):
        """Update the footprint options based on the selected type."""
        fp_type = self.fp_type_var.get()
        
        # Clear the options frame
        for widget in self.fp_options_frame.winfo_children():
            widget.destroy()
        
        # Show options based on footprint type
        if fp_type == "smd_resistor":
            self._show_smd_resistor_options()
        elif fp_type == "smd_capacitor":
            self._show_smd_capacitor_options()
        elif fp_type == "sot23":
            self._show_sot23_options()
        elif fp_type == "dip":
            self._show_dip_options()
        elif fp_type == "header":
            self._show_header_options()
    
    def _show_smd_resistor_options(self):
        """Show options for SMD resistor footprint."""
        # SMD resistor options frame
        smd_res_frame = ttk.LabelFrame(self.fp_options_frame, text="SMD Resistor Options")
        smd_res_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Size
        ttk.Label(smd_res_frame, text="Size:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.smd_res_size_var = tk.StringVar(value="0805")
        size_combo = ttk.Combobox(smd_res_frame, textvariable=self.smd_res_size_var, width=10)
        size_combo["values"] = ["0402", "0603", "0805", "1206", "1210", "2010", "2512"]
        size_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Auto-populate name based on size
        ttk.Label(smd_res_frame, text="Naming Convention:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.smd_res_naming_var = tk.StringVar(value="R_${SIZE}")
        ttk.Entry(smd_res_frame, textvariable=self.smd_res_naming_var).grid(row=1, column=1, columnspan=2, sticky=tk.W+tk.E, padx=5, pady=2)
        ttk.Label(smd_res_frame, text="(${SIZE} will be replaced with the size)").grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=5, pady=0)
        
        # Set initial name
        if not self.fp_name_var.get():
            self.fp_name_var.set(f"R_{self.smd_res_size_var.get()}")
            self.fp_desc_var.set(f"Resistor SMD {self.smd_res_size_var.get()}, reflow soldering")
            self.fp_tags_var.set(f"resistor {self.smd_res_size_var.get()}")
    
    def _show_smd_capacitor_options(self):
        """Show options for SMD capacitor footprint."""
        # SMD capacitor options frame
        smd_cap_frame = ttk.LabelFrame(self.fp_options_frame, text="SMD Capacitor Options")
        smd_cap_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Size
        ttk.Label(smd_cap_frame, text="Size:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.smd_cap_size_var = tk.StringVar(value="0805")
        size_combo = ttk.Combobox(smd_cap_frame, textvariable=self.smd_cap_size_var, width=10)
        size_combo["values"] = ["0402", "0603", "0805", "1206", "1210", "2010", "2512"]
        size_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Polarized
        self.smd_cap_polarized_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(smd_cap_frame, text="Polarized", variable=self.smd_cap_polarized_var).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        
        # Auto-populate name based on size and polarized state
        ttk.Label(smd_cap_frame, text="Naming Convention:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.smd_cap_naming_var = tk.StringVar(value="C_${SIZE}")
        ttk.Entry(smd_cap_frame, textvariable=self.smd_cap_naming_var).grid(row=2, column=1, columnspan=2, sticky=tk.W+tk.E, padx=5, pady=2)
        ttk.Label(smd_cap_frame, text="(${SIZE} will be replaced with the size)").grid(row=3, column=1, columnspan=2, sticky=tk.W, padx=5, pady=0)
        
        # Set initial name
        if not self.fp_name_var.get():
            self.fp_name_var.set(f"C_{self.smd_cap_size_var.get()}")
            self.fp_desc_var.set(f"Capacitor SMD {self.smd_cap_size_var.get()}, reflow soldering")
            self.fp_tags_var.set(f"capacitor {self.smd_cap_size_var.get()}")
    
    def _show_sot23_options(self):
        """Show options for SOT-23 footprint."""
        # SOT-23 options frame
        sot_frame = ttk.LabelFrame(self.fp_options_frame, text="SOT-23 Options")
        sot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Number of pins
        ttk.Label(sot_frame, text="Number of Pins:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.sot_pins_var = tk.IntVar(value=3)
        pins_combo = ttk.Combobox(sot_frame, textvariable=self.sot_pins_var, width=5)
        pins_combo["values"] = [3, 5, 6]
        pins_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Set initial name
        if not self.fp_name_var.get():
            self.fp_name_var.set(f"SOT-23-{self.sot_pins_var.get()}")
            self.fp_desc_var.set(f"SOT-23, {self.sot_pins_var.get()} Pin package")
            self.fp_tags_var.set(f"SOT-23 SOT-23-{self.sot_pins_var.get()}")
    
    def _show_dip_options(self):
        """Show options for DIP footprint."""
        # DIP options frame
        dip_frame = ttk.LabelFrame(self.fp_options_frame, text="DIP Options")
        dip_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Number of pins
        ttk.Label(dip_frame, text="Number of Pins:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.dip_pins_var = tk.IntVar(value=8)
        pins_combo = ttk.Combobox(dip_frame, textvariable=self.dip_pins_var, width=5)
        pins_combo["values"] = [4, 6, 8, 14, 16, 18, 20, 24, 28, 32, 40, 48]
        pins_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Width
        ttk.Label(dip_frame, text="Width (mm):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.dip_width_var = tk.DoubleVar(value=7.62)
        width_combo = ttk.Combobox(dip_frame, textvariable=self.dip_width_var, width=5)
        width_combo["values"] = [7.62, 10.16, 15.24]
        width_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Pitch
        ttk.Label(dip_frame, text="Pitch (mm):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.dip_pitch_var = tk.DoubleVar(value=2.54)
        ttk.Entry(dip_frame, textvariable=self.dip_pitch_var, width=5).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Set initial name
        if not self.fp_name_var.get():
            width_in = round(self.dip_width_var.get() / 2.54, 2)
            width_str = f"{width_in:.1f}".rstrip('0').rstrip('.') if width_in == int(width_in) else f"{width_in:.2f}"
            self.fp_name_var.set(f"DIP-{self.dip_pins_var.get()}_W{width_str}in")
            self.fp_desc_var.set(f"DIP-{self.dip_pins_var.get()}, {width_str}\" wide package")
            self.fp_tags_var.set(f"DIP DIL PDIP {self.dip_pins_var.get()} {width_str}\"")
    
    def _show_header_options(self):
        """Show options for header connector footprint."""
        # Header options frame
        header_frame = ttk.LabelFrame(self.fp_options_frame, text="Header Connector Options")
        header_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Number of pins
        ttk.Label(header_frame, text="Number of Pins:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.header_pins_var = tk.IntVar(value=8)
        ttk.Spinbox(header_frame, from_=1, to=100, textvariable=self.header_pins_var, width=5).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Number of rows
        ttk.Label(header_frame, text="Number of Rows:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.header_rows_var = tk.IntVar(value=1)
        rows_combo = ttk.Combobox(header_frame, textvariable=self.header_rows_var, width=5)
        rows_combo["values"] = [1, 2]
        rows_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Pitch
        ttk.Label(header_frame, text="Pitch (mm):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.header_pitch_var = tk.DoubleVar(value=2.54)
        ttk.Entry(header_frame, textvariable=self.header_pitch_var, width=5).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Set initial name
        if not self.fp_name_var.get():
            rows = self.header_rows_var.get()
            pins = self.header_pins_var.get()
            pitch = self.header_pitch_var.get()
            pitch_str = f"{pitch:.2f}mm".rstrip('0').rstrip('.')
            
            if rows == 1:
                self.fp_name_var.set(f"PinHeader_1x{pins}_{pitch_str}_Straight")
                self.fp_desc_var.set(f"Pin header 1x{pins}, straight, {pitch_str} pitch")
                self.fp_tags_var.set(f"pin header {pins}x1 {pitch_str}")
            else:
                cols = pins // rows
                self.fp_name_var.set(f"PinHeader_{rows}x{cols}_{pitch_str}_Straight")
                self.fp_desc_var.set(f"Pin header {rows}x{cols}, straight, {pitch_str} pitch")
                self.fp_tags_var.set(f"pin header {rows}x{cols} {pitch_str}")
    
    def _populate_sym_lib_list(self):
        """Populate the symbol library listbox."""
        self.sym_lib_listbox.delete(0, tk.END)
        for lib_name in sorted(self.lib_manager.symbol_libs.keys()):
            self.sym_lib_listbox.insert(tk.END, lib_name)
    
    def _populate_fp_lib_list(self):
        """Populate the footprint library listbox."""
        self.fp_lib_listbox.delete(0, tk.END)
        for lib_name in sorted(self.lib_manager.fp_libs.keys()):
            self.fp_lib_listbox.insert(tk.END, lib_name)
    
    def _add_symbol_library(self):
        """Add a new symbol library."""
        lib_name = simpledialog.askstring("New Library", "Enter library name:")
        if lib_name:
            try:
                self.lib_manager.create_symbol_lib(lib_name)
                self._populate_sym_lib_list()
                self.symbol_lib_combo["values"] = list(self.lib_manager.symbol_libs.keys())
                self.status_var.set(f"Created symbol library: {lib_name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create library: {e}")
    
    def _add_footprint_library(self):
        """Add a new footprint library."""
        lib_name = simpledialog.askstring("New Library", "Enter library name:")
        if lib_name:
            try:
                self.lib_manager.create_footprint_lib(lib_name)
                self._populate_fp_lib_list()
                self.fp_lib_combo["values"] = list(self.lib_manager.fp_libs.keys())
                self.status_var.set(f"Created footprint library: {lib_name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create library: {e}")
    
    def _create_symbol(self):
        """Create a symbol based on the current settings."""
        symbol_type = self.symbol_type_var.get()
        name = self.symbol_name_var.get()
        lib_name = self.symbol_lib_var.get()
        reference = self.symbol_ref_var.get()
        description = self.symbol_desc_var.get()
        keywords = self.symbol_keywords_var.get().split()
        
        if not name:
            messagebox.showerror("Error", "Symbol name is required.")
            return
        
        if not lib_name:
            messagebox.showerror("Error", "Please select a library.")
            return
        
        try:
            # Create the symbol based on type
            if symbol_type == "ic":
                left_pins = self.left_pins_var.get()
                right_pins = self.right_pins_var.get()
                top_pins = self.top_pins_var.get()
                bottom_pins = self.bottom_pins_var.get()
                width = self.ic_width_var.get()
                height = self.ic_height_var.get()
                
                symbol = SymbolGenerator.create_ic(
                    name=name,
                    reference=reference,
                    left_pins=left_pins,
                    right_pins=right_pins,
                    top_pins=top_pins,
                    bottom_pins=bottom_pins,
                    width=width,
                    height=height
                )
            
            elif symbol_type == "resistor":
                style = self.res_style_var.get()
                symbol = SymbolGenerator.create_resistor(
                    name=name,
                    reference=reference,
                    style=style.lower()
                )
                
                # Add power rating to description if provided
                if self.res_power_var.get():
                    description += f", {self.res_power_var.get()}W"
                
                # Add tolerance to description if provided
                if self.res_tolerance_var.get():
                    description += f", ±{self.res_tolerance_var.get()}%"
            
            elif symbol_type == "capacitor":
                style = self.cap_style_var.get()
                symbol = SymbolGenerator.create_capacitor(
                    name=name,
                    reference=reference,
                    polarized=(style == "POLARIZED"),
                    variable=(style == "VARIABLE")
                )
                
                # Add voltage rating to description if provided
                if self.cap_voltage_var.get():
                    description += f", {self.cap_voltage_var.get()}V"
                
                # Add tolerance to description if provided
                if self.cap_tolerance_var.get():
                    description += f", ±{self.cap_tolerance_var.get()}%"
            
            elif symbol_type == "connector":
                pins = self.conn_pins_var.get()
                style = self.conn_style_var.get()
                
                if style == "SINGLE_ROW":
                    symbol = SymbolGenerator.create_connector(
                        name=name, 
                        reference=reference,
                        pins=pins,
                        rows=1
                    )
                elif style == "DOUBLE_ROW":
                    symbol = SymbolGenerator.create_connector(
                        name=name, 
                        reference=reference,
                        pins=pins,
                        rows=2
                    )
                else:  # GRID
                    # Calculate a reasonable grid for the pins
                    grid_dim = int(math.sqrt(pins))
                    symbol = SymbolGenerator.create_connector(
                        name=name, 
                        reference=reference,
                        pins=pins,
                        rows=grid_dim
                    )
            
            # Set common properties
            symbol.description = description
            symbol.keywords = keywords
            
            # Save the symbol
            self.symbol_writer.write(symbol, lib_name)
            
            self.status_var.set(f"Created symbol '{name}' in library '{lib_name}'")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create symbol: {e}")
            raise
    
    def _create_footprint(self):
        """Create a footprint based on the current settings."""
        fp_type = self.fp_type_var.get()
        name = self.fp_name_var.get()
        lib_name = self.fp_lib_var.get()
        description = self.fp_desc_var.get()
        tags = self.fp_tags_var.get().split()
        
        if not name:
            messagebox.showerror("Error", "Footprint name is required.")
            return
        
        if not lib_name:
            messagebox.showerror("Error", "Please select a library.")
            return
        
        try:
            # Create the footprint based on type
            if fp_type == "smd_resistor":
                size = self.smd_res_size_var.get()
                footprint = FootprintGenerator.create_smd_resistor(
                    name=name,
                    size=size,
                    description=description,
                    tags=tags
                )
            
            elif fp_type == "smd_capacitor":
                size = self.smd_cap_size_var.get()
                polarized = self.smd_cap_polarized_var.get()
                footprint = FootprintGenerator.create_smd_capacitor(
                    name=name,
                    size=size,
                    polarized=polarized,
                    description=description,
                    tags=tags
                )
                
            elif fp_type == "sot23":
                pins = self.sot_pins_var.get()
                footprint = FootprintGenerator.create_sot23(
                    name=name,
                    pins=pins,
                    description=description,
                    tags=tags
                )
                
            elif fp_type == "dip":
                pins = self.dip_pins_var.get()
                width = self.dip_width_var.get()
                pitch = self.dip_pitch_var.get()
                footprint = FootprintGenerator.create_dip(
                    name=name,
                    pins=pins,
                    width=width,
                    pitch=pitch,
                    description=description,
                    tags=tags
                )
                
            elif fp_type == "header":
                pins = self.header_pins_var.get()
                rows = self.header_rows_var.get()
                pitch = self.header_pitch_var.get()
                footprint = FootprintGenerator.create_connector_header(
                    name=name,
                    pins=pins,
                    rows=rows,
                    pitch=pitch,
                    description=description,
                    tags=tags
              footprint = FootprintGenerator.create_connector_header(
                    name=name,
                    pins=pins,
                    rows=rows,
                    pitch=pitch,
                    description=description,
                    tags=tags
                )
            
            # Save the footprint
            self.footprint_writer.write(footprint, lib_name)
            
            self.status_var.set(f"Created footprint '{name}' in library '{lib_name}'")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create footprint: {e}")
            raise
    
    def _create_about_tab(self):
        """Create the about tab content."""
        about_frame = ttk.Frame(self.notebook)
        self.notebook.add(about_frame, text="About")
        
        # App info
        info_frame = ttk.LabelFrame(about_frame, text="KiCad Symbol and Footprint Creator")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(info_frame, text="Version 1.0", font=('Helvetica', 10)).pack(pady=5)
        ttk.Label(info_frame, text="A tool for creating KiCad symbols and footprints").pack(pady=5)
        
        # Credits
        credits_frame = ttk.LabelFrame(about_frame, text="Credits")
        credits_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(credits_frame, text="Developed by: Your Name", font=('Helvetica', 10)).pack(pady=5)
        ttk.Label(credits_frame, text="Using: Python, Tkinter, and KiCad libraries", font=('Helvetica', 10)).pack(pady=5)
        
        # Help links
        help_frame = ttk.LabelFrame(about_frame, text="Help Resources")
        help_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        def open_url(url):
            import webbrowser
            webbrowser.open_new(url)
        
        kicad_link = ttk.Label(help_frame, text="KiCad Website", cursor="hand2", foreground="blue")
        kicad_link.pack(pady=5)
        kicad_link.bind("<Button-1>", lambda e: open_url("https://www.kicad.org/"))
        
        docs_link = ttk.Label(help_frame, text="KiCad Documentation", cursor="hand2", foreground="blue")
        docs_link.pack(pady=5)
        docs_link.bind("<Button-1>", lambda e: open_url("https://docs.kicad.org/"))
        
        github_link = ttk.Label(help_frame, text="Project Repository", cursor="hand2", foreground="blue")
        github_link.pack(pady=5)
        github_link.bind("<Button-1>", lambda e: open_url("https://github.com/yourusername/kicad-symbol-footprint-creator"))

    def _create_batch_tab(self):
        """Create the batch processing tab content."""
        batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(batch_frame, text="Batch Processing")
        
        # File selection
        file_frame = ttk.LabelFrame(batch_frame, text="Input File")
        file_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(file_frame, text="CSV or JSON file with component definitions:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.batch_file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.batch_file_var, width=40).grid(row=0, column=1, padx=5, pady=5)
        
        def browse_file():
            filename = filedialog.askopenfilename(
                filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json"), ("All files", "*.*")]
            )
            if filename:
                self.batch_file_var.set(filename)
        
        ttk.Button(file_frame, text="Browse...", command=browse_file).grid(row=0, column=2, padx=5, pady=5)
        
        # Output options
        output_frame = ttk.LabelFrame(batch_frame, text="Output Options")
        output_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Symbol library
        ttk.Label(output_frame, text="Symbol Library:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.batch_sym_lib_var = tk.StringVar()
        sym_lib_combo = ttk.Combobox(output_frame, textvariable=self.batch_sym_lib_var, width=20)
        sym_lib_combo["values"] = list(self.lib_manager.symbol_libs.keys())
        sym_lib_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Footprint library
        ttk.Label(output_frame, text="Footprint Library:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.batch_fp_lib_var = tk.StringVar()
        fp_lib_combo = ttk.Combobox(output_frame, textvariable=self.batch_fp_lib_var, width=20)
        fp_lib_combo["values"] = list(self.lib_manager.fp_libs.keys())
        fp_lib_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Process options
        process_frame = ttk.LabelFrame(batch_frame, text="Process Options")
        process_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.create_symbols_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(process_frame, text="Create Symbols", variable=self.create_symbols_var).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.create_footprints_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(process_frame, text="Create Footprints", variable=self.create_footprints_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.associate_fp_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(process_frame, text="Associate Footprints to Symbols", variable=self.associate_fp_var).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Log output
        log_frame = ttk.LabelFrame(batch_frame, text="Processing Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(batch_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Process Batch", command=self._process_batch).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Clear Log", command=lambda: self.log_text.delete(1.0, tk.END)).pack(side=tk.RIGHT, padx=5)
    
    def _process_batch(self):
        """Process a batch file of component definitions."""
        filename = self.batch_file_var.get()
        
        if not filename:
            messagebox.showerror("Error", "Please select an input file.")
            return
        
        if not os.path.exists(filename):
            messagebox.showerror("Error", f"File not found: {filename}")
            return
        
        sym_lib = self.batch_sym_lib_var.get()
        fp_lib = self.batch_fp_lib_var.get()
        
        if self.create_symbols_var.get() and not sym_lib:
            messagebox.showerror("Error", "Please select a symbol library.")
            return
        
        if self.create_footprints_var.get() and not fp_lib:
            messagebox.showerror("Error", "Please select a footprint library.")
            return
        
        try:
            # Load components from file
            components = self._load_components_from_file(filename)
            
            self.log_text.insert(tk.END, f"Loaded {len(components)} components from {filename}\n")
            
            # Process each component
            for i, comp in enumerate(components):
                self.log_text.insert(tk.END, f"\nProcessing component {i+1}: {comp.get('name', 'Unnamed')}\n")
                
                # Create symbol if requested
                if self.create_symbols_var.get():
                    try:
                        self._create_component_symbol(comp, sym_lib)
                        self.log_text.insert(tk.END, f"  - Created symbol in library {sym_lib}\n")
                    except Exception as e:
                        self.log_text.insert(tk.END, f"  - Failed to create symbol: {e}\n")
                
                # Create footprint if requested
                if self.create_footprints_var.get():
                    try:
                        self._create_component_footprint(comp, fp_lib)
                        self.log_text.insert(tk.END, f"  - Created footprint in library {fp_lib}\n")
                    except Exception as e:
                        self.log_text.insert(tk.END, f"  - Failed to create footprint: {e}\n")
                
                # Associate footprint if requested
                if self.associate_fp_var.get() and self.create_symbols_var.get() and self.create_footprints_var.get():
                    try:
                        self._associate_footprint_to_symbol(comp, sym_lib, fp_lib)
                        self.log_text.insert(tk.END, f"  - Associated footprint to symbol\n")
                    except Exception as e:
                        self.log_text.insert(tk.END, f"  - Failed to associate footprint: {e}\n")
            
            self.log_text.insert(tk.END, "\nBatch processing completed.\n")
            self.log_text.see(tk.END)
            
            self.status_var.set(f"Processed {len(components)} components from {os.path.basename(filename)}")
            
        except Exception as e:
            self.log_text.insert(tk.END, f"Error processing batch: {e}\n")
            messagebox.showerror("Error", f"Failed to process batch: {e}")
    
    def _load_components_from_file(self, filename):
        """Load component definitions from a CSV or JSON file."""
        components = []
        
        # Determine file type from extension
        _, ext = os.path.splitext(filename)
        
        if ext.lower() == '.json':
            # Load JSON file
            with open(filename, 'r') as f:
                components = json.load(f)
        
        elif ext.lower() == '.csv':
            # Load CSV file
            with open(filename, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    components.append(row)
        
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        return components
    
    def _create_component_symbol(self, comp, lib_name):
        """Create a symbol from component definition."""
        # Extract symbol properties
        name = comp.get('name', '')
        symbol_type = comp.get('symbol_type', 'ic')
        reference = comp.get('reference', 'U')
        description = comp.get('description', '')
        keywords = comp.get('keywords', '').split()
        
        # Create symbol based on type
        if symbol_type == 'resistor':
            style = comp.get('resistor_style', 'iec')
            symbol = SymbolGenerator.create_resistor(
                name=name,
                reference=reference,
                style=style.lower()
            )
        
        elif symbol_type == 'capacitor':
            polarized = comp.get('capacitor_polarized', 'false').lower() == 'true'
            variable = comp.get('capacitor_variable', 'false').lower() == 'true'
            symbol = SymbolGenerator.create_capacitor(
                name=name,
                reference=reference,
                polarized=polarized,
                variable=variable
            )
        
        elif symbol_type == 'connector':
            pins = int(comp.get('connector_pins', 8))
            rows = int(comp.get('connector_rows', 1))
            symbol = SymbolGenerator.create_connector(
                name=name,
                reference=reference,
                pins=pins,
                rows=rows
            )
        
        else:  # Default to IC
            left_pins = int(comp.get('ic_left_pins', 0))
            right_pins = int(comp.get('ic_right_pins', 0))
            top_pins = int(comp.get('ic_top_pins', 0))
            bottom_pins = int(comp.get('ic_bottom_pins', 0))
            width = float(comp.get('ic_width', 20.0))
            height = float(comp.get('ic_height', 20.0))
            
            symbol = SymbolGenerator.create_ic(
                name=name,
                reference=reference,
                left_pins=left_pins,
                right_pins=right_pins,
                top_pins=top_pins,
                bottom_pins=bottom_pins,
                width=width,
                height=height
            )
        
        # Set common properties
        symbol.description = description
        symbol.keywords = keywords
        
        # Save the symbol
        self.symbol_writer.write(symbol, lib_name)
        
        return symbol
    
    def _create_component_footprint(self, comp, lib_name):
        """Create a footprint from component definition."""
        # Extract footprint properties
        name = comp.get('footprint_name', comp.get('name', ''))
        footprint_type = comp.get('footprint_type', 'smd_resistor')
        description = comp.get('footprint_description', '')
        tags = comp.get('footprint_tags', '').split()
        
        # Create footprint based on type
        if footprint_type == 'smd_resistor':
            size = comp.get('smd_resistor_size', '0805')
            footprint = FootprintGenerator.create_smd_resistor(
                name=name,
                size=size,
                description=description,
                tags=tags
            )
        
        elif footprint_type == 'smd_capacitor':
            size = comp.get('smd_capacitor_size', '0805')
            polarized = comp.get('smd_capacitor_polarized', 'false').lower() == 'true'
            footprint = FootprintGenerator.create_smd_capacitor(
                name=name,
                size=size,
                polarized=polarized,
                description=description,
                tags=tags
            )
        
        elif footprint_type == 'sot23':
            pins = int(comp.get('sot23_pins', 3))
            footprint = FootprintGenerator.create_sot23(
                name=name,
                pins=pins,
                description=description,
                tags=tags
            )
        
        elif footprint_type == 'dip':
            pins = int(comp.get('dip_pins', 8))
            width = float(comp.get('dip_width', 7.62))
            pitch = float(comp.get('dip_pitch', 2.54))
            footprint = FootprintGenerator.create_dip(
                name=name,
                pins=pins,
                width=width,
                pitch=pitch,
                description=description,
                tags=tags
            )
        
        elif footprint_type == 'header':
            pins = int(comp.get('header_pins', 8))
            rows = int(comp.get('header_rows', 1))
            pitch = float(comp.get('header_pitch', 2.54))
            footprint = FootprintGenerator.create_connector_header(
                name=name,
                pins=pins,
                rows=rows,
                pitch=pitch,
                description=description,
                tags=tags
            )
        
        else:
            raise ValueError(f"Unsupported footprint type: {footprint_type}")
        
        # Save the footprint
        self.footprint_writer.write(footprint, lib_name)
        
        return footprint
    
    def _associate_footprint_to_symbol(self, comp, sym_lib, fp_lib):
        """Associate a footprint to a symbol."""
        symbol_name = comp.get('name', '')
        footprint_name = comp.get('footprint_name', symbol_name)
        
        # Create the footprint association string
        fp_association = f"{fp_lib}:{footprint_name}"
        
        # Update the symbol's footprint field
        self.lib_manager.set_symbol_footprint(sym_lib, symbol_name, fp_association)
    
    def run(self):
        """Run the application main loop."""
        self.mainloop()


if __name__ == "__main__":
    app = KiCadComponentCreator()
    app.run()
