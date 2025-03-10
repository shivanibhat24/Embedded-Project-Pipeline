import sys
import os
import json
import threading
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import pytesseract
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks

# Import the component generator class from the previous file
from kicad_component_generator import KiCadComponentGenerator

class KiCadComponentGeneratorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("KiCad Component Generator")
        self.root.geometry("1000x700")
        self.root.minsize(900, 600)
        
        self.sensor_image_path = None
        self.pin_diagram_path = None
        self.generator = KiCadComponentGenerator()
        
        self.config = self.load_config()
        
        self.setup_ui()
    
    def load_config(self):
        """Load or create configuration file."""
        config_path = "config.json"
        default_config = {
            "output_dir": "output",
            "temp_dir": "temp",
            "tesseract_path": "",
            "kicad_lib_path": "",
            "recent_files": [],
            "component_database": "component_database.json",
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults for any missing keys
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
        except FileNotFoundError:
            config = default_config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
        
        # Create necessary directories
        os.makedirs(config["output_dir"], exist_ok=True)
        os.makedirs(config["temp_dir"], exist_ok=True)
        
        return config
    
    def save_config(self):
        """Save current configuration to file."""
        with open("config.json", 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def setup_ui(self):
        """Set up the main user interface."""
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header frame
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(header_frame, text="KiCad Component Generator", font=("Arial", 16, "bold")).pack(side=tk.LEFT)
        
        ttk.Button(header_frame, text="Settings", command=self.open_settings).pack(side=tk.RIGHT)
        ttk.Button(header_frame, text="Help", command=self.show_help).pack(side=tk.RIGHT, padx=5)
        
        # Create content notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.import_tab = ttk.Frame(self.notebook)
        self.preview_tab = ttk.Frame(self.notebook)
        self.output_tab = ttk.Frame(self.notebook)
        self.database_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.import_tab, text="Import Images")
        self.notebook.add(self.preview_tab, text="Preview & Edit")
        self.notebook.add(self.output_tab, text="Generated Files")
        self.notebook.add(self.database_tab, text="Component Database")
        
        # Set up each tab
        self.setup_import_tab()
        self.setup_preview_tab()
        self.setup_output_tab()
        self.setup_database_tab()
        
        # Create status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
        
        # Set up progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
        self.progress_bar.pack_forget()  # Hide initially
    
    def setup_import_tab(self):
        """Set up the Import Images tab."""
        # Create two columns
        left_frame = ttk.LabelFrame(self.import_tab, text="Sensor Image")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        right_frame = ttk.LabelFrame(self.import_tab, text="Pin Diagram")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Sensor image frame
        self.sensor_canvas = tk.Canvas(left_frame, bg="white", bd=2, relief=tk.SUNKEN)
        self.sensor_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        sensor_btn_frame = ttk.Frame(left_frame)
        sensor_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(sensor_btn_frame, text="Browse...", command=self.browse_sensor_image).pack(side=tk.LEFT)
        ttk.Button(sensor_btn_frame, text="Capture...", command=lambda: self.capture_image("sensor")).pack(side=tk.LEFT, padx=5)
        ttk.Button(sensor_btn_frame, text="Clear", command=lambda: self.clear_image("sensor")).pack(side=tk.LEFT)
        
        # Pin diagram frame
        self.pin_canvas = tk.Canvas(right_frame, bg="white", bd=2, relief=tk.SUNKEN)
        self.pin_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        pin_btn_frame = ttk.Frame(right_frame)
        pin_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(pin_btn_frame, text="Browse...", command=self.browse_pin_diagram).pack(side=tk.LEFT)
        ttk.Button(pin_btn_frame, text="Capture...", command=lambda: self.capture_image("pin")).pack(side=tk.LEFT, padx=5)
        ttk.Button(pin_btn_frame, text="Clear", command=lambda: self.clear_image("pin")).pack(side=tk.LEFT)
        
        # Bottom frame for processing
        bottom_frame = ttk.Frame(self.import_tab)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=10)
        
        ttk.Button(bottom_frame, text="Process Images", command=self.process_images).pack(side=tk.RIGHT)
        ttk.Label(bottom_frame, text="Step 1: Import or capture sensor image and pin diagram").pack(side=tk.LEFT)
    
    def setup_preview_tab(self):
        """Set up the Preview & Edit tab."""
        # Create left panel for component info
        left_frame = ttk.LabelFrame(self.preview_tab, text="Component Information")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create info form
        form_frame = ttk.Frame(left_frame)
        form_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Part name
        ttk.Label(form_frame, text="Part Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.part_name_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.part_name_var).grid(row=0, column=1, sticky=tk.EW, pady=5, padx=5)
        
        # Part description
        ttk.Label(form_frame, text="Description:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.part_desc_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.part_desc_var).grid(row=1, column=1, sticky=tk.EW, pady=5, padx=5)
        
        # Part shape
        ttk.Label(form_frame, text="Shape:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.part_shape_var = tk.StringVar()
        shape_combo = ttk.Combobox(form_frame, textvariable=self.part_shape_var)
        shape_combo['values'] = ('rectangular', 'circular', 'polygon', 'irregular')
        shape_combo.grid(row=2, column=1, sticky=tk.EW, pady=5, padx=5)
        
        # Package type
        ttk.Label(form_frame, text="Package:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.package_var = tk.StringVar()
        package_combo = ttk.Combobox(form_frame, textvariable=self.package_var)
        package_combo['values'] = ('DIP', 'SOIC', 'QFP', 'QFN', 'BGA', 'TO', 'Custom')
        package_combo.grid(row=3, column=1, sticky=tk.EW, pady=5, padx=5)
        
        # Dimensions
        dim_frame = ttk.LabelFrame(form_frame, text="Dimensions (mm)")
        dim_frame.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=5)
        
        ttk.Label(dim_frame, text="Width:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        self.width_var = tk.DoubleVar(value=0.0)
        ttk.Entry(dim_frame, textvariable=self.width_var, width=10).grid(row=0, column=1, sticky=tk.W, pady=5, padx=5)
        
        ttk.Label(dim_frame, text="Height:").grid(row=0, column=2, sticky=tk.W, pady=5, padx=5)
        self.height_var = tk.DoubleVar(value=0.0)
        ttk.Entry(dim_frame, textvariable=self.height_var, width=10).grid(row=0, column=3, sticky=tk.W, pady=5, padx=5)
        
        ttk.Label(dim_frame, text="Thickness:").grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        self.thickness_var = tk.DoubleVar(value=1.0)
        ttk.Entry(dim_frame, textvariable=self.thickness_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=5, padx=5)
        
        form_frame.columnconfigure(1, weight=1)
        
        # Create right panel for pin information
        right_frame = ttk.LabelFrame(self.preview_tab, text="Pin Information")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create pin table
        table_frame = ttk.Frame(right_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Table headers
        columns = ('number', 'name', 'type', 'position')
        self.pin_tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        
        self.pin_tree.heading('number', text='Pin #')
        self.pin_tree.heading('name', text='Name')
        self.pin_tree.heading('type', text='Type')
        self.pin_tree.heading('position', text='Position')
        
        self.pin_tree.column('number', width=50)
        self.pin_tree.column('name', width=150)
        self.pin_tree.column('type', width=100)
        self.pin_tree.column('position', width=100)
        
        self.pin_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.pin_tree.yview)
        self.pin_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Button frame for pin editing
        pin_btn_frame = ttk.Frame(right_frame)
        pin_btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(pin_btn_frame, text="Add Pin", command=self.add_pin).pack(side=tk.LEFT)
        ttk.Button(pin_btn_frame, text="Edit Pin", command=self.edit_pin).pack(side=tk.LEFT, padx=5)
        ttk.Button(pin_btn_frame, text="Delete Pin", command=self.delete_pin).pack(side=tk.LEFT)
        
        # Bottom frame for processing
        bottom_frame = ttk.Frame(self.preview_tab)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=10)
        
        ttk.Button(bottom_frame, text="Generate KiCad Files", command=self.generate_kicad_files).pack(side=tk.RIGHT)
        ttk.Label(bottom_frame, text="Step 2: Review and edit component information").pack(side=tk.LEFT)
    
    def setup_output_tab(self):
        """Set up the Generated Files tab."""
        # Create left panel for file list
        left_frame = ttk.LabelFrame(self.output_tab, text="Generated Files")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # File list
        self.file_list = tk.Listbox(left_frame, selectmode=tk.SINGLE)
        self.file_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.file_list.bind('<<ListboxSelect>>', self.show_file_preview)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.file_list.yview)
        self.file_list.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create right panel for file preview
        right_frame = ttk.LabelFrame(self.output_tab, text="File Preview")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # File preview text area
        self.file_preview = tk.Text(right_frame, wrap=tk.NONE, bg="white", fg="black")
        self.file_preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbars
        yscrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.file_preview.yview)
        self.file_preview.configure(yscroll=yscrollbar.set)
        yscrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        xscrollbar = ttk.Scrollbar(right_frame, orient=tk.HORIZONTAL, command=self.file_preview.xview)
        self.file_preview.configure(xscroll=xscrollbar.set)
        xscrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bottom frame for actions
        bottom_frame = ttk.Frame(self.output_tab)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=10)
        
        ttk.Button(bottom_frame, text="Open in KiCad", command=self.open_in_kicad).pack(side=tk.RIGHT)
        ttk.Button(bottom_frame, text="Open Output Folder", command=self.open_output_folder).pack(side=tk.RIGHT, padx=5)
        ttk.Button(bottom_frame, text="Save to Library", command=self.save_to_library).pack(side=tk.RIGHT, padx=5)
        ttk.Label(bottom_frame, text="Step 3: Review and use generated files").pack(side=tk.LEFT)
    
    def setup_database_tab(self):
        """Set up the Component Database tab."""
        # Create top frame for search and filters
        top_frame = ttk.Frame(self.database_tab)
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(top_frame, text="Search:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(top_frame, textvariable=self.search_var, width=30)
        search_entry.pack(side=tk.LEFT, padx=5)
        search_entry.bind("<Return>", lambda e: self.search_database())
        
        ttk.Button(top_frame, text="Search", command=self.search_database).pack(side=tk.LEFT)
        
        ttk.Label(top_frame, text="Category:").pack(side=tk.LEFT, padx=(20, 5))
        self.category_var = tk.StringVar()
        category_combo = ttk.Combobox(top_frame, textvariable=self.category_var, width=15)
        category_combo['values'] = ('All', 'Sensors', 'Microcontrollers', 'ICs', 'Connectors', 'Passive')
        category_combo.current(0)
        category_combo.pack(side=tk.LEFT)
        category_combo.bind("<<ComboboxSelected>>", lambda e: self.search_database())
        
        # Create main frame for database table
        main_frame = ttk.Frame(self.database_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Database table
        columns = ('name', 'description', 'package', 'pins', 'category')
        self.db_tree = ttk.Treeview(main_frame, columns=columns, show='headings')
        
        self.db_tree.heading('name', text='Component Name')
        self.db_tree.heading('description', text='Description')
        self.db_tree.heading('package', text='Package')
        self.db_tree.heading('pins', text='Pins')
        self.db_tree.heading('category', text='Category')
        
        self.db_tree.column('name', width=150)
        self.db_tree.column('description', width=250)
        self.db_tree.column('package', width=100)
        self.db_tree.column('pins', width=50)
        self.db_tree.column('category', width=100)
        
        self.db_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.db_tree.bind("<Double-1>", self.load_component_from_db)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.db_tree.yview)
        self.db_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bottom frame for actions
        bottom_frame = ttk.Frame(self.database_tab)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=10)
        
        ttk.Button(bottom_frame, text="Load Component", command=self.load_selected_component).pack(side=tk.RIGHT)
        ttk.Button(bottom_frame, text="Delete", command=self.delete_from_database).pack(side=tk.RIGHT, padx=5)
        ttk.Button(bottom_frame, text="Import Database", command=self.import_database).pack(side=tk.RIGHT, padx=5)
        ttk.Button(bottom_frame, text="Export Database", command=self.export_database).pack(side=tk.RIGHT, padx=5)
        
        # Load database
        self.load_database()
    
    def browse_sensor_image(self):
        """Browse for sensor image file."""
        file_path = filedialog.askopenfilename(
            title="Select Sensor Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        
        if file_path:
            self.sensor_image_path = file_path
            self.display_image(self.sensor_canvas, file_path)
            self.status_var.set(f"Sensor image loaded: {os.path.basename(file_path)}")
    
    def browse_pin_diagram(self):
        """Browse for pin diagram file."""
        file_path = filedialog.askopenfilename(
            title="Select Pin Diagram",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        
        if file_path:
            self.pin_diagram_path = file_path
            self.display_image(self.pin_canvas, file_path)
            self.status_var.set(f"Pin diagram loaded: {os.path.basename(file_path)}")
    
    def display_image(self, canvas, image_path):
        """Display image on the specified canvas."""
        try:
            # Clear canvas
            canvas.delete("all")
            
            # Load and resize image
            pil_img = Image.open(image_path)
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            # Resize to fit canvas while maintaining aspect ratio
            img_width, img_height = pil_img.size
            ratio = min(canvas_width/img_width, canvas_height/img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage and display
            img = ImageTk.PhotoImage(pil_img)
            canvas.img = img  # Keep reference to prevent garbage collection
            canvas.create_image(canvas_width // 2, canvas_height // 2, image=img)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not display image: {str(e)}")
    
    def capture_image(self, image_type):
        """Capture image from camera."""
        def capture_thread():
            try:
                # Initialize camera
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    messagebox.showerror("Error", "Could not open camera")
                    return
                
                # Create window for preview
                cv2.namedWindow(f"Capture {image_type.capitalize()} Image", cv2.WINDOW_NORMAL)
                cv2.resizeWindow(f"Capture {image_type.capitalize()} Image", 640, 480)
                
                while True:
                    # Capture frame
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Display frame
                    cv2.imshow(f"Capture {image_type.capitalize()} Image", frame)
                    
                    # Check for key press
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC key
                        break
                    elif key == 32:  # Spacebar
                        # Save captured image
                        timestamp = int(time.time())
                        file_path = os.path.join(self.config["temp_dir"], f"{image_type}_{timestamp}.jpg")
                        cv2.imwrite(file_path, frame)
                        
                        # Set path and display
                        if image_type == "sensor":
                            self.sensor_image_path = file_path
                            self.root.after(100, lambda: self.display_image(self.sensor_canvas, file_path))
                        else:
                            self.pin_diagram_path = file_path
                            self.root.after(100, lambda: self.display_image(self.pin_canvas, file_path))
                        
                        self.root.after(100, lambda: self.status_var.set(f"{image_type.capitalize()} image captured"))
                        break
                
                # Release resources
                cap.release()
                cv2.destroyAllWindows()
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Error capturing image: {str(e)}"))
        
        # Run in separate thread to avoid freezing UI
        import time
        thread = threading.Thread(target=capture_thread)
        thread.daemon = True
        thread.start()
    
    def clear_image(self, image_type):
        """Clear the selected image."""
        if image_type == "sensor":
            self.sensor_image_path = None
            self.sensor_canvas.delete("all")
        else:
            self.pin_diagram_path = None
            self.pin_canvas.delete("all")
        
        self.status_var.set(f"{image_type.capitalize()} image cleared")
    
    def process_images(self):
        """Process the loaded images to extract component information."""
        if not self.sensor_image_path or not self.pin_diagram_path:
            messagebox.showerror("Error", "Please load both sensor image and pin diagram")
            return
        
        try:
            self.progress_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
            self.progress_var.set(0)
            self.status_var.set("Processing images...")
            
            def process_thread():
                try:
                    # Update progress
                    self.root.after(0, lambda: self.progress_var.set(10))
                    
                    # Process images using the component generator
                    component_info = self.generator.process_image(self.sensor_image_path)
                    self.root.after(0, lambda: self.progress_var.set(50))
                    
                    pins = self.generator.process_pin_diagram(self.pin_diagram_path)
                    self.root.after(0, lambda: self.progress_var.set(90))
                    
                    # Update UI with extracted information
                    self.root.after(0, lambda: self.update_preview_tab(component_info, pins))
                    self.root.after(0, lambda: self.progress_var.set(100))
                    self.root.after(0, lambda: self.status_var.set("Processing complete"))
                    self.root.after(500, lambda: self.progress_bar.pack_forget())
                    
                    # Switch to preview tab
                    self.root.after(0, lambda: self.notebook.select(self.preview_tab))
                    
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Error processing images: {str(e)}"))
                    self.root.after(0, lambda: self.status_var.set("Error processing images"))
                    self.root.after(0, lambda: self.progress_bar.pack_forget())
            
            # Run in separate thread to avoid freezing UI
            thread = threading.Thread(target=process_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")
            self.progress_bar.pack_forget()
    
    def update_preview_tab(self, component_info, pins):
        """Update the preview tab with processed information."""
        # Update component info
        part_name = component_info.get("part_info", {}).get("part_number", "SENSOR")
        self.part_name_var.set(part_name)
        self.part_desc_var.set(f"{part_name} sensor")
        self.part_shape_var.set(component_info["shape"])
        
        # Set default package based on shape
        if component_info["shape"] == "rectangular":
            self.package_var.set("SOIC")
        else:
            self.package_var.set("Custom")
        
        # Set dimensions
        width = component_info["dimensions"]["width"]
        height = component_info["dimensions"]["height"]
        
        # Convert pixel dimensions to mm (approximate)
        scale_factor = 0.1  # This is an approximation
        self.width_var.set(round(width * scale_factor, 2))
        self.height_var.set(round(height * scale_factor, 2))
        
        # Clear and update pin table
        for item in self.pin_tree.get_children():
            self.pin_tree.delete(item)
        
        # Add pins to table
        for i, pin in enumerate(pins):
            pin_number = pin.get("number", str(i+1))
            pin_name = pin.get("name", f"PIN{i+1}")
            pin_type = "I/O"  # Default type
            pin_position = pin.get("position", "-")
            
            if isinstance(pin_position, tuple):
                pin_position = f"({pin_position[0]}, {pin_position[1]})"
            
            self.pin_tree.insert('', 'end', values=(pin_number, pin_name, pin_type, pin_position))
    
    def add_pin(self):
        """Add a new pin to the pin table."""
        # Create dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Pin")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Create form
        ttk.Label(dialog, text="Pin Number:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        pin_num_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=pin_num_var).grid(row=0, column=1, sticky=tk.EW, padx=10, pady=5)
        
        ttk.Label(dialog, text="Pin Name:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        pin_name_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=pin_name_var).grid(row=1, column=1, sticky=tk.EW, padx=10, pady=5)
        
        ttk.Label(dialog, text="Pin Type:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        pin_type_var = tk.StringVar(value="I/O")
        type_combo = ttk.Combobox(dialog, textvariable=pin_type_var, width=15)
        type_combo['values'] = ('I/O', 'Input', 'Output', 'Power', 'GND', 'NC')
        type_combo.grid(row=2, column=1, sticky=tk.EW, padx=10, pady=5)
        
        ttk.Label(dialog, text="Position:").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
        pin_pos_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=pin_pos_var).grid(row=3, column=1, sticky=tk.EW, padx=10, pady=5)
        
        # Button frame
        btn_frame = ttk.Frame(dialog)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=15)
        
        # Add button action
        def add_pin_action():
            # Add pin to treeview
            self.pin_tree.insert('', 'end', values=(
                pin_num_var.get(), 
                pin_name_var.get(), 
                pin_type_var.get(), 
                pin_pos_var.get()
            ))
            dialog.destroy()
        
        ttk.Button(btn_frame, text="Add", command=add_pin_action).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Make dialog resizable
        dialog.columnconfigure(1, weight=1)
        
        # Center dialog
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (self.root.winfo_width() // 2) - (width // 2) + self.root.winfo_x()
        y = (self.root.winfo_height() // 2) - (height // 2) + self.root.winfo_y()
        dialog.geometry(f"+{x}+{y}")
        
        # Set focus
        pin_num_var.set(str(len(self.pin_tree.get_children()) + 1))
        dialog.focus_set()
    
    def edit_pin(self):
        """Edit the selected pin."""
        # Get selected item
        selected = self.pin_tree.selection()
        if not selected:
            messagebox.showinfo("Info", "Please select a pin to edit")
            return
        
        item = selected[0]
        values = self.pin_tree.item(item, 'values')
        
        # Create dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Pin")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Create form
        ttk.Label(dialog, text="Pin Number:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        pin_num_var = tk.StringVar(value=values[0])
        ttk.Entry(dialog, textvariable=pin_num_var).grid(row=0, column=1, sticky=tk.EW, padx=10, pady=5)
        
        ttk.Label(dialog, text="Pin Name:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        pin_name_var = tk.StringVar(value=values[1])
        ttk.Entry(dialog, textvariable=pin_name_var).grid(row=1, column=1, sticky=tk.EW, padx=10, pady=5)
        
        ttk.Label(dialog, text="Pin Type:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        pin_type_var = tk.StringVar(value=values[2])
        type_combo = ttk.Combobox(dialog, textvariable=pin_type_var, width=15)
        type_combo['values'] = ('I/O', 'Input', 'Output', 'Power', 'GND', 'NC')
        type_combo.grid(row=2, column=1, sticky=tk.EW, padx=10, pady=5)
        
        ttk.Label(dialog, text="Position:").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
        pin_pos_var = tk.StringVar(value=values[3])
        ttk.Entry(dialog, textvariable=pin_pos_var).grid(row=3, column=1, sticky=tk.EW, padx=10, pady=5)
        
        # Button frame
        btn_frame = ttk.Frame(dialog)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=15)
        
        # Update button action
        def update_pin_action():
            # Update pin in treeview
            self.pin_tree.item(item, values=(
                pin_num_var.get(), 
                pin_name_var.get(), 
                pin_type_var.get(), 
                pin_pos_var.get()
            ))
            dialog.destroy()
        
        ttk.Button(btn_frame, text="Update", command=update_pin_action).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Make dialog resizable
        dialog.columnconfigure(1, weight=1)
        
        # Center dialog
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (self.root.winfo_width() // 2) - (width // 2) + self.root.winfo_x()
        y = (self.root.winfo_height() // 2) - (height // 2) + self.root.winfo_y()
        dialog.geometry(f"+{x}+{y}")
        
        dialog.focus_set()
    
    def delete_pin(self):
        """Delete the selected pin."""
        selected = self.pin_tree.selection()
        if not selected:
            messagebox.showinfo("Info", "Please select a pin to delete")
            return
        
        confirm = messagebox.askyesno("Confirm", "Are you sure you want to delete this pin?")
        if confirm:
            self.pin_tree.delete(selected)
            self.status_var.set("Pin deleted")
    
    def generate_kicad_files(self):
        """Generate KiCad files from the component information."""
        # Get component information
        component = {
            "name": self.part_name_var.get(),
            "description": self.part_desc_var.get(),
            "shape": self.part_shape_var.get(),
            "package": self.package_var.get(),
            "dimensions": {
                "width": self.width_var.get(),
                "height": self.height_var.get(),
                "thickness": self.thickness_var.get()
            },
            "pins": []
        }
        
        # Get pins from treeview
        for item in self.pin_tree.get_children():
            values = self.pin_tree.item(item, 'values')
            pin = {
                "number": values[0],
                "name": values[1],
                "type": values[2],
                "position": values[3]
            }
            component["pins"].append(pin)
        
        if not component["name"]:
            messagebox.showerror("Error", "Please enter a component name")
            return
        
        if not component["pins"]:
            messagebox.showerror("Error", "Please add at least one pin")
            return
        
        try:
            self.progress_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
            self.progress_var.set(0)
            self.status_var.set("Generating KiCad files...")
            
            def generate_thread():
                try:
                    # Update progress
                    self.root.after(0, lambda: self.progress_var.set(10))
                    
                    # Generate KiCad files
                    output_files = self.generator.generate_kicad_files(component, self.config["output_dir"])
                    self.root.after(0, lambda: self.progress_var.set(90))
                    
                    # Update UI with generated files
                    self.root.after(0, lambda: self.update_output_tab(output_files))
                    self.root.after(0, lambda: self.progress_var.set(100))
                    self.root.after(0, lambda: self.status_var.set("Generation complete"))
                    self.root.after(500, lambda: self.progress_bar.pack_forget())
                    
                    # Switch to output tab
                    self.root.after(0, lambda: self.notebook.select(self.output_tab))
                    
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Error generating files: {str(e)}"))
                    self.root.after(0, lambda: self.status_var.set("Error generating files"))
                    self.root.after(0, lambda: self.progress_bar.pack_forget())
            
            # Run in separate thread to avoid freezing UI
            thread = threading.Thread(target=generate_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")
            self.progress_bar.pack_forget()
    
    def update_output_tab(self, output_files):
        """Update the output tab with generated files."""
        # Clear file list
        self.file_list.delete(0, tk.END)
        
        # Add files to list
        for file_path in output_files:
            self.file_list.insert(tk.END, os.path.basename(file_path))
        
        # Select first file
        if self.file_list.size() > 0:
            self.file_list.selection_set(0)
            self.show_file_preview(None)
    
    def show_file_preview(self, event):
        """Show preview of the selected file."""
        try:
            # Get selected file
            selected = self.file_list.curselection()
            if not selected:
                return
            
            file_name = self.file_list.get(selected[0])
            file_path = os.path.join(self.config["output_dir"], file_name)
            
            # Clear preview
            self.file_preview.delete(1.0, tk.END)
            
            # Read and display file content
            with open(file_path, 'r') as f:
                content = f.read()
                self.file_preview.insert(tk.END, content)
            
        except Exception as e:
            self.file_preview.delete(1.0, tk.END)
            self.file_preview.insert(tk.END, f"Error loading file: {str(e)}")
    
    def open_in_kicad(self):
        """Open the selected file in KiCad."""
        selected = self.file_list.curselection()
        if not selected:
            messagebox.showinfo("Info", "Please select a file to open")
            return
        
        file_name = self.file_list.get(selected[0])
        file_path = os.path.join(self.config["output_dir"], file_name)
        
        kicad_path = self.config.get("kicad_lib_path", "")
        if not kicad_path:
            kicad_path = filedialog.askdirectory(title="Select KiCad installation directory")
            if kicad_path:
                self.config["kicad_lib_path"] = kicad_path
                self.save_config()
            else:
                return
        
        try:
            # Determine which KiCad application to open based on file extension
            ext = os.path.splitext(file_name)[1].lower()
            app_name = ""
            
            if ext == ".lib" or ext == ".kicad_sym":
                app_name = "eeschema"
            elif ext == ".mod" or ext == ".kicad_mod":
                app_name = "pcbnew"
            elif ext == ".3dshapes":
                app_name = "kicad"
            else:
                app_name = "kicad"
            
            # Construct command
            if sys.platform.startswith('win'):
                app_path = os.path.join(kicad_path, f"{app_name}.exe")
                cmd = [app_path, file_path]
            elif sys.platform.startswith('darwin'):  # macOS
                app_path = os.path.join(kicad_path, f"{app_name}.app")
                cmd = ["open", "-a", app_path, file_path]
            else:  # Linux
                cmd = [app_name, file_path]
            
            # Execute command
            subprocess.Popen(cmd)
            self.status_var.set(f"Opening {file_name} in KiCad")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error opening file in KiCad: {str(e)}")
    
    def open_output_folder(self):
        """Open the output folder in file explorer."""
        output_dir = os.path.abspath(self.config["output_dir"])
        
        try:
            if sys.platform.startswith('win'):
                os.startfile(output_dir)
            elif sys.platform.startswith('darwin'):  # macOS
                subprocess.Popen(["open", output_dir])
            else:  # Linux
                subprocess.Popen(["xdg-open", output_dir])
                
            self.status_var.set(f"Opened output folder: {output_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error opening output folder: {str(e)}")
    
    def save_to_library(self):
        """Save component to the database."""
        component = {
            "name": self.part_name_var.get(),
            "description": self.part_desc_var.get(),
            "package": self.package_var.get(),
            "shape": self.part_shape_var.get(),
            "dimensions": {
                "width": self.width_var.get(),
                "height": self.height_var.get(),
                "thickness": self.thickness_var.get()
            },
            "pins": []
        }
        
        # Get pins from treeview
        for item in self.pin_tree.get_children():
            values = self.pin_tree.item(item, 'values')
            pin = {
                "number": values[0],
                "name": values[1],
                "type": values[2],
                "position": values[3]
            }
            component["pins"].append(pin)
        
        # Create dialog for additional info
        dialog = tk.Toplevel(self.root)
        dialog.title("Save to Library")
        dialog.geometry("350x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Component Name:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        name_var = tk.StringVar(value=component["name"])
        ttk.Entry(dialog, textvariable=name_var).grid(row=0, column=1, sticky=tk.EW, padx=10, pady=5)
        
        ttk.Label(dialog, text="Category:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        category_var = tk.StringVar(value="Sensors")
        category_combo = ttk.Combobox(dialog, textvariable=category_var, width=15)
        category_combo['values'] = ('Sensors', 'Microcontrollers', 'ICs', 'Connectors', 'Passive', 'Other')
        category_combo.grid(row=1, column=1, sticky=tk.EW, padx=10, pady=5)
        
        ttk.Label(dialog, text="Manufacturer:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        manufacturer_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=manufacturer_var).grid(row=2, column=1, sticky=tk.EW, padx=10, pady=5)
        
        # Button frame
        btn_frame = ttk.Frame(dialog)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=15)
        
        def save_action():
            component["name"] = name_var.get()
            component["category"] = category_var.get()
            component["manufacturer"] = manufacturer_var.get()
            
            if not component["name"]:
                messagebox.showerror("Error", "Please enter a component name")
                return
            
            try:
                # Save to database
                self.save_component_to_db(component)
                messagebox.showinfo("Success", "Component saved to library")
                dialog.destroy()
                
                # Refresh database tab
                self.load_database()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error saving component: {str(e)}")
        
        ttk.Button(btn_frame, text="Save", command=save_action).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Make dialog resizable
        dialog.columnconfigure(1, weight=1)
        
        # Center dialog
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (self.root.winfo_width() // 2) - (width // 2) + self.root.winfo_x()
        y = (self.root.winfo_height() // 2) - (height // 2) + self.root.winfo_y()
        dialog.geometry(f"+{x}+{y}")
        
        dialog.focus_set()
    
    def load_database(self):
        """Load component database."""
        try:
            db_path = self.config["component_database"]
            
            # Clear database tree
            for item in self.db_tree.get_children():
                self.db_tree.delete(item)
            
            # Load database
            if os.path.exists(db_path):
                with open(db_path, 'r') as f:
                    database = json.load(f)
            else:
                database = []
                with open(db_path, 'w') as f:
                    json.dump(database, f, indent=4)
            
            # Populate database tree
            for component in database:
                self.db_tree.insert('', 'end', values=(
                    component["name"],
                    component.get("description", ""),
                    component.get("package", ""),
                    len(component.get("pins", [])),
                    component.get("category", "")
                ))
            
            self.status_var.set(f"Loaded {len(database)} components from database")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading database: {str(e)}")
    
    def save_component_to_db(self, component):
        """Save component to database."""
        db_path = self.config["component_database"]
        
        # Load existing database
        if os.path.exists(db_path):
            with open(db_path, 'r') as f:
                database = json.load(f)
        else:
            database = []
        
        # Check if component already exists
        for i, existing in enumerate(database):
            if existing["name"] == component["name"]:
                # Ask for confirmation to overwrite
                confirm = messagebox.askyesno(
                    "Confirm", 
                    f"Component '{component['name']}' already exists. Overwrite?"
                )
                if confirm:
                    database[i] = component
                    break
                else:
                    return
        else:
            # Add new component
            database.append(component)
        
        # Save database
        with open(db_path, 'w') as f:
            json.dump(database, f, indent=4)
    
    def search_database(self):
        """Search the component database."""
        try:
            search_text = self.search_var.get().lower()
            category = self.category_var.get()
            
            # Clear database tree
            for item in self.db_tree.get_children():
                self.db_tree.delete(item)
            
            # Load database
            db_path = self.config["component_database"]
            if os.path.exists(db_path):
                with open(db_path, 'r') as f:
                    database = json.load(f)
            else:
                database = []
            
            # Filter by search text and category
            results = []
            for component in database:
                if category != "All" and component.get("category", "") != category:
                    continue
                
                if search_text:
                    # Search in name, description, and manufacturer
                    if (search_text in component["name"].lower() or
                        search_text in component.get("description", "").lower() or
                        search_text in component.get("manufacturer", "").lower()):
                        results.append(component)
                else:
                    results.append(component)
            
            # Populate database tree
            for component in results:
                self.db_tree.insert('', 'end', values=(
                    component["name"],
                    component.get("description", ""),
                    component.get("package", ""),
                    len(component.get("pins", [])),
                    component.get("category", "")
                ))
            
            self.status_var.set(f"Found {len(results)} components")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error searching database: {str(e)}")
    
    def load_component_from_db(self, event):
        """Load component from database on double-click."""
        self.load_selected_component()
    
    def load_selected_component(self):
        """Load the selected component from database."""
        selected = self.db_tree.selection()
        if not selected:
            messagebox.showinfo("Info", "Please select a component to load")
            return
        
        try:
            # Get component name
            values = self.db_tree.item(selected[0], 'values')
            component_name = values[0]
            
            # Load database
            db_path = self.config["component_database"]
            with open(db_path, 'r') as f:
                database = json.load(f)
            
            # Find component
            component = None
            for item in database:
                if item["name"] == component_name:
                    component = item
                    break
            
            if not component:
                messagebox.showerror("Error", f"Component '{component_name}' not found in database")
                return
            
            # Update UI with component information
            self.part_name_var.set(component["name"])
            self.part_desc_var.set(component.get("description", ""))
            self.part_shape_var.set(component.get("shape", "rectangular"))
            self.package_var.set(component.get("package", "SOIC"))
            
            dimensions = component.get("dimensions", {})
            self.width_var.set(dimensions.get("width", 0.0))
            self.height_var.set(dimensions.get("height", 0.0))
            self.thickness_var.set(dimensions.get("thickness", 1.0))
            
            # Clear and update pin table
            for item in self.pin_tree.get_children():
                self.pin_tree.delete(item)
            
            # Add pins to table
            for pin in component.get("pins", []):
                pin_number = pin.get("number", "")
                pin_name = pin.get("name", "")
                pin_type = pin.get("type", "I/O")
                pin_position = pin.get("position", "")
                
                self.pin_tree.insert('', 'end', values=(pin_number, pin_name, pin_type, pin_position))
            
            # Switch to preview tab
            self.notebook.select(self.preview_tab)
            self.status_var.set(f"Loaded component: {component_name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading component: {str(e)}")
    
    def delete_from_database(self):
        """Delete selected component from database."""
        selected = self.db_tree.selection()
        if not selected:
            messagebox.showinfo("Info", "Please select a component to delete")
            return
        
        values = self.db_tree.item(selected[0], 'values')
        component_name = values[0]
        
        confirm = messagebox.askyesno(
            "Confirm", 
            f"Are you sure you want to delete component '{component_name}' from the database?"
        )
        if not confirm:
            return
        
        try:
            # Load database
            db_path = self.config["component_database"]
            with open(db_path, 'r') as f:
                database = json.load(f)
            
            # Remove component
            database = [item for item in database if item["name"] != component_name]
            
            # Save database
            with open(db_path, 'w') as f:
                json.dump(database, f, indent=4)
            
            # Remove from tree
            self.db_tree.delete(selected)
            self.status_var.set(f"Deleted component: {component_name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error deleting component: {str(e)}")
    
    def import_database(self):
        """Import component database from a JSON file."""
        file_path = filedialog.askopenfilename(
            title="Import Database",
            filetypes=[("JSON files", "*.json")],
            initialdir=os.path.dirname(self.config["component_database"])
        )
        
        if not file_path:
            return
        
        try:
            # Load imported database
            with open(file_path, 'r') as f:
                imported_db = json.load(f)
            
            # Load current database
            db_path = self.config["component_database"]
            if os.path.exists(db_path):
                with open(db_path, 'r') as f:
                    current_db = json.load(f)
            else:
                current_db = []
            
            # Ask for merge or replace
            if current_db:
                option = messagebox.askyesnocancel(
                    "Import Options",
                    "Do you want to merge with the existing database?\n"
                    "Yes - Merge (will overwrite existing components with the same name)\n"
                    "No - Replace (will delete the current database and replace with imported)\n"
                    "Cancel - Abort operation"
                )
                
                if option is None:  # Cancel
                    return
                elif option:  # Yes - Merge
                    # Create a dictionary for faster lookup
                    current_dict = {item["name"]: i for i, item in enumerate(current_db)}
                    
                    # Merge databases
                    for item in imported_db:
                        if item["name"] in current_dict:
                            # Replace existing component
                            current_db[current_dict[item["name"]]] = item
                        else:
                            # Add new component
                            current_db.append(item)
                    
                    final_db = current_db
                else:  # No - Replace
                    final_db = imported_db
            else:
                final_db = imported_db
            
            # Save merged or replaced database
            with open(db_path, 'w') as f:
                json.dump(final_db, f, indent=4)
            
            # Reload database
            self.load_database()
            
            messagebox.showinfo("Import Complete", f"Imported {len(imported_db)} components")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error importing database: {str(e)}")
    
    def export_database(self):
        """Export component database to a JSON file."""
        file_path = filedialog.asksaveasfilename(
            title="Export Database",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialdir=os.path.dirname(self.config["component_database"])
        )
        
        if not file_path:
            return
        
        try:
            # Load current database
            db_path = self.config["component_database"]
            if os.path.exists(db_path):
                with open(db_path, 'r') as f:
                    database = json.load(f)
            else:
                database = []
            
            # Save to specified file
            with open(file_path, 'w') as f:
                json.dump(database, f, indent=4)
            
            messagebox.showinfo("Export Complete", f"Exported {len(database)} components to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting database: {str(e)}")
    
    def open_settings(self):
        """Open settings dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Settings")
        dialog.geometry("500x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Create settings form
        ttk.Label(dialog, text="Output Directory:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        output_var = tk.StringVar(value=self.config["output_dir"])
        output_entry = ttk.Entry(dialog, textvariable=output_var, width=30)
        output_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(dialog, text="Browse...", command=lambda: self.browse_dir(output_var)).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(dialog, text="KiCad Library Path:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        kicad_var = tk.StringVar(value=self.config.get("kicad_lib_path", ""))
        kicad_entry = ttk.Entry(dialog, textvariable=kicad_var, width=30)
        kicad_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(dialog, text="Browse...", command=lambda: self.browse_dir(kicad_var)).grid(row=1, column=2, padx=5, pady=5)
        
        ttk.Label(dialog, text="Component Database:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        db_var = tk.StringVar(value=self.config["component_database"])
        db_entry = ttk.Entry(dialog, textvariable=db_var, width=30)
        db_entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(dialog, text="Browse...", command=lambda: self.browse_file(db_var, ".json")).grid(row=2, column=2, padx=5, pady=5)
        
        ttk.Label(dialog, text="Default Shape:").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
        shape_var = tk.StringVar(value=self.config.get("default_shape", "rectangular"))
        shape_combo = ttk.Combobox(dialog, textvariable=shape_var, width=15)
        shape_combo['values'] = ('rectangular', 'circular', 'custom')
        shape_combo.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(dialog, text="Default Package:").grid(row=4, column=0, sticky=tk.W, padx=10, pady=5)
        package_var = tk.StringVar(value=self.config.get("default_package", "SOIC"))
        package_combo = ttk.Combobox(dialog, textvariable=package_var, width=15)
        package_combo['values'] = ('DIP', 'SOIC', 'SSOP', 'QFP', 'QFN', 'BGA', 'TO', 'SOT')
        package_combo.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Button frame
        btn_frame = ttk.Frame(dialog)
        btn_frame.grid(row=5, column=0, columnspan=3, pady=15)
        
        def save_settings():
            # Update config
            self.config["output_dir"] = output_var.get()
            self.config["kicad_lib_path"] = kicad_var.get()
            self.config["component_database"] = db_var.get()
            self.config["default_shape"] = shape_var.get()
            self.config["default_package"] = package_var.get()
            
            # Save config
            self.save_config()
            dialog.destroy()
            
            # Update UI
            self.status_var.set("Settings saved")
        
        ttk.Button(btn_frame, text="Save", command=save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Make dialog resizable
        dialog.columnconfigure(1, weight=1)
        
        # Center dialog
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (self.root.winfo_width() // 2) - (width // 2) + self.root.winfo_x()
        y = (self.root.winfo_height() // 2) - (height // 2) + self.root.winfo_y()
        dialog.geometry(f"+{x}+{y}")
        
        dialog.focus_set()
    
    def browse_dir(self, var):
        """Browse for directory."""
        directory = filedialog.askdirectory()
        if directory:
            var.set(directory)
    
    def browse_file(self, var, ext=None):
        """Browse for file."""
        filetypes = [("All Files", "*.*")]
        if ext:
            filetypes.insert(0, (f"{ext.upper()} Files", f"*{ext}"))
        
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        if file_path:
            var.set(file_path)
    
    def save_config(self):
        """Save configuration to file."""
        try:
            # Create config directory if it doesn't exist
            config_dir = os.path.dirname(self.config_file)
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            # Save config
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error saving configuration: {str(e)}")
    
    def show_help(self):
        """Show help information."""
        help_text = """
KiCAD Component Generator Help

This application helps you create custom components for KiCAD.

Getting Started:
1. Fill in the component details in the Preview tab
2. Add pins using the Add button
3. Click Generate to create KiCAD files
4. The generated files will appear in the Output tab
5. You can save components to the library for future use

Keyboard Shortcuts:
- Ctrl+N: New component
- Ctrl+G: Generate KiCAD files
- Ctrl+S: Save to library
- F1: Show this help
- Esc: Exit

"""
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Help")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        
        # Text widget with scrollbar
        frame = ttk.Frame(dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text = tk.Text(frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text.insert(tk.END, help_text)
        text.config(state=tk.DISABLED)
        
        scrollbar.config(command=text.yview)
        
        ttk.Button(dialog, text="OK", command=dialog.destroy).pack(pady=10)
        
        # Center dialog
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (self.root.winfo_width() // 2) - (width // 2) + self.root.winfo_x()
        y = (self.root.winfo_height() // 2) - (height // 2) + self.root.winfo_y()
        dialog.geometry(f"+{x}+{y}")
    
    def show_about(self):
        """Show about dialog."""
        about_text = """
KiCAD Component Generator
Version 1.0.0

A tool for creating custom components for KiCAD electronic design software.

Features:
- Create schematic symbols
- Generate footprints
- Create 3D models
- Manage component library

© 2025 Your Name
License: MIT
"""
        
        dialog = tk.Toplevel(self.root)
        dialog.title("About")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        
        # Logo or icon (placeholder)
        canvas = tk.Canvas(dialog, width=100, height=100)
        canvas.pack(pady=10)
        canvas.create_oval(10, 10, 90, 90, fill="lightblue")
        canvas.create_text(50, 50, text="KCG", font=("Arial", 20, "bold"))
        
        # Text
        label = ttk.Label(dialog, text=about_text, justify=tk.CENTER)
        label.pack(padx=20, pady=10)
        
        # Close button
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
        
        # Center dialog
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (self.root.winfo_width() // 2) - (width // 2) + self.root.winfo_x()
        y = (self.root.winfo_height() // 2) - (height // 2) + self.root.winfo_y()
        dialog.geometry(f"+{x}+{y}")
    
    def exit_application(self):
        """Exit the application."""
        confirm = messagebox.askyesno("Exit", "Are you sure you want to exit?")
        if confirm:
            self.root.destroy()
    
    def create_preview_canvas(self):
        """Create a canvas for component preview."""
        preview_frame = ttk.LabelFrame(self.preview_tab, text="Component Preview")
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.preview_canvas = tk.Canvas(preview_frame, bg="white", width=300, height=300)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add canvas controls
        control_frame = ttk.Frame(preview_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Zoom:").pack(side=tk.LEFT, padx=5)
        zoom_var = tk.DoubleVar(value=1.0)
        zoom_scale = ttk.Scale(control_frame, from_=0.5, to=3.0, variable=zoom_var, orient=tk.HORIZONTAL)
        zoom_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Update preview when component changes
        def update_preview(*args):
            self.draw_component_preview()
        
        # Bind to variables
        self.part_shape_var.trace_add("write", update_preview)
        self.width_var.trace_add("write", update_preview)
        self.height_var.trace_add("write", update_preview)
        zoom_var.trace_add("write", update_preview)
        
        # Bind to tree changes
        self.pin_tree.bind("<<TreeviewSelect>>", update_preview)
        
        # Initialize preview
        self.draw_component_preview()
    
    def draw_component_preview(self):
        """Draw component preview on canvas."""
        try:
            # Clear canvas
            self.preview_canvas.delete("all")
            
            # Get component information
            shape = self.part_shape_var.get()
            width = float(self.width_var.get()) if self.width_var.get() else 0
            height = float(self.height_var.get()) if self.height_var.get() else 0
            
            # Get canvas dimensions
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
            # Calculate scale and center position
            scale = min(canvas_width / 200, canvas_height / 200) * 0.8
            center_x = canvas_width / 2
            center_y = canvas_height / 2
            
            # Draw component outline
            if shape == "rectangular":
                # Draw rectangle
                x1 = center_x - (width * scale / 2)
                y1 = center_y - (height * scale / 2)
                x2 = center_x + (width * scale / 2)
                y2 = center_y + (height * scale / 2)
                self.preview_canvas.create_rectangle(x1, y1, x2, y2, outline="black", width=2, fill="white")
            elif shape == "circular":
                # Draw circle
                radius = max(width, height) * scale / 2
                x1 = center_x - radius
                y1 = center_y - radius
                x2 = center_x + radius
                y2 = center_y + radius
                self.preview_canvas.create_oval(x1, y1, x2, y2, outline="black", width=2, fill="white")
            else:
                # Draw default shape
                self.preview_canvas.create_rectangle(center_x-50, center_y-50, center_x+50, center_y+50, 
                                               outline="black", width=2, fill="white")
            
            # Draw pins
            pin_length = 20
            pins = []
            
            # Get pins from treeview
            for item in self.pin_tree.get_children():
                values = self.pin_tree.item(item, 'values')
                pins.append({
                    "number": values[0],
                    "name": values[1],
                    "type": values[2],
                    "position": values[3]
                })
            
            # Draw pins
            for pin in pins:
                pin_pos = pin.get("position", "").upper()
                pin_num = pin.get("number", "")
                pin_name = pin.get("name", "")
                pin_type = pin.get("type", "I/O")
                
                # Determine pin position
                if pin_pos == "LEFT" or pin_pos == "L":
                    start_x = center_x - (width * scale / 2)
                    end_x = start_x - pin_length
                    pin_y = center_y - (height * scale / 4) + (int(pin_num) % 10) * (height * scale / 5)
                    self.preview_canvas.create_line(start_x, pin_y, end_x, pin_y, fill="black", width=2)
                    self.preview_canvas.create_text(end_x - 5, pin_y - 10, text=pin_num, anchor=tk.E)
                    self.preview_canvas.create_text(start_x + 5, pin_y, text=pin_name, anchor=tk.W)
                elif pin_pos == "RIGHT" or pin_pos == "R":
                    start_x = center_x + (width * scale / 2)
                    end_x = start_x + pin_length
                    pin_y = center_y - (height * scale / 4) + (int(pin_num) % 10) * (height * scale / 5)
                    self.preview_canvas.create_line(start_x, pin_y, end_x, pin_y, fill="black", width=2)
                    self.preview_canvas.create_text(end_x + 5, pin_y - 10, text=pin_num, anchor=tk.W)
                    self.preview_canvas.create_text(start_x - 5, pin_y, text=pin_name, anchor=tk.E)
                elif pin_pos == "TOP" or pin_pos == "T":
                    start_y = center_y - (height * scale / 2)
                    end_y = start_y - pin_length
                    pin_x = center_x - (width * scale / 4) + (int(pin_num) % 10) * (width * scale / 5)
                    self.preview_canvas.create_line(pin_x, start_y, pin_x, end_y, fill="black", width=2)
                    self.preview_canvas.create_text(pin_x - 5, end_y - 5, text=pin_num, anchor=tk.E)
                    self.preview_canvas.create_text(pin_x, start_y - 5, text=pin_name, anchor=tk.S)
                elif pin_pos == "BOTTOM" or pin_pos == "B":
                    start_y = center_y + (height * scale / 2)
                    end_y = start_y + pin_length
                    pin_x = center_x - (width * scale / 4) + (int(pin_num) % 10) * (width * scale / 5)
                    self.preview_canvas.create_line(pin_x, start_y, pin_x, end_y, fill="black", width=2)
                    self.preview_canvas.create_text(pin_x - 5, end_y + 5, text=pin_num, anchor=tk.E)
                    self.preview_canvas.create_text(pin_x, start_y + 5, text=pin_name, anchor=tk.N)
                else:
                    # Default to right side
                    start_x = center_x + (width * scale / 2)
                    end_x = start_x + pin_length
                    pin_y = center_y - (height * scale / 4) + (int(pin_num) % 10) * (height * scale / 5)
                    self.preview_canvas.create_line(start_x, pin_y, end_x, pin_y, fill="black", width=2)
                    self.preview_canvas.create_text(end_x + 5, pin_y - 10, text=pin_num, anchor=tk.W)
                    self.preview_canvas.create_text(start_x - 5, pin_y, text=pin_name, anchor=tk.E)
                
                # Draw pin type indicator
                if pin_type == "Input":
                    # Draw input arrow
                    if pin_pos in ["LEFT", "L"]:
                        self.preview_canvas.create_polygon(start_x-5, pin_y-5, start_x, pin_y, start_x-5, pin_y+5, fill="black")
                elif pin_type == "Output":
                    # Draw output arrow
                    if pin_pos in ["RIGHT", "R"]:
                        self.preview_canvas.create_polygon(start_x+5, pin_y-5, start_x, pin_y, start_x+5, pin_y+5, fill="black")
                elif pin_type == "Power":
                    # Draw power symbol (P)
                    self.preview_canvas.create_text(start_x if pin_pos in ["TOP", "BOTTOM", "T", "B"] else pin_y, 
                                                   start_y if pin_pos in ["LEFT", "RIGHT", "L", "R"] else pin_x, 
                                                   text="P", font=("Arial", 8, "bold"))
                elif pin_type == "GND":
                    # Draw ground symbol
                    self.preview_canvas.create_text(start_x if pin_pos in ["TOP", "BOTTOM", "T", "B"] else pin_y, 
                                                   start_y if pin_pos in ["LEFT", "RIGHT", "L", "R"] else pin_x, 
                                                   text="⏚", font=("Arial", 12))
            
            # Draw component name
            component_name = self.part_name_var.get()
            if component_name:
                self.preview_canvas.create_text(center_x, center_y - 10, text=component_name, font=("Arial", 12, "bold"))
            
            # Draw package name
            package_name = self.package_var.get()
            if package_name:
                self.preview_canvas.create_text(center_x, center_y + 10, text=package_name, font=("Arial", 10))
            
        except Exception as e:
            # Ignore preview errors
            pass


# Main entry point
def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='kicad_component_generator.log'
    )
    
    # Create root window
    root = tk.Tk()
    root.title("KiCAD Component Generator")
    root.geometry("1000x700")
    
    # Set icon (placeholder)
    # root.iconbitmap('icon.ico')
    
    # Apply theme
    style = ttk.Style()
    if sys.platform.startswith('win'):
        style.theme_use('vista')
    elif sys.platform.startswith('darwin'):  # macOS
        style.theme_use('aqua')
    else:  # Linux/Unix
        style.theme_use('clam')
    
    # Initialize generator
    generator = KiCadGenerator()
    
    # Create UI
    app = ComponentGeneratorUI(root, generator)
    
    # Start main loop
    root.mainloop()


if __name__ == "__main__":
    main()
