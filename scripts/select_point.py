# create a UI to select a point from an image
import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class PointSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("Point Selector")
        
        # Initialize variables
        self.image = None
        self.photo = None
        self.current_point = None
        self.point_coordinates = None
        self.video_path = None
        self.current_frame = None
        self.point_marker = None
        self.marker_size = 5  # Size of the point marker
        
        # Create UI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Create buttons frame
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Create input type selection
        self.input_type = ttk.Combobox(
            self.button_frame, 
            values=["Image", "Video"],
            state="readonly",
            width=10
        )
        self.input_type.set("Image")
        self.input_type.pack(side=tk.LEFT, padx=5)
        
        # Load button
        self.load_btn = tk.Button(self.button_frame, text="Load File", command=self.load_file)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        # Save coordinates button
        self.save_btn = tk.Button(self.button_frame, text="Save Coordinates", command=self.save_coordinates)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Clear point button
        self.clear_btn = tk.Button(self.button_frame, text="Clear Point", command=self.clear_point)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Create canvas
        self.canvas = tk.Canvas(self.root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events - only need click event for point selection
        self.canvas.bind("<Button-1>", self.on_click)
        
    def load_file(self):
        if self.input_type.get() == "Image":
            self.load_image()
        else:
            self.load_video()

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        
        if file_path:
            self.image = cv2.imread(file_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.current_frame = self.image.copy()
            self.display_image()
            
    def load_video(self):
        self.video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        
        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_frame = self.image.copy()
                self.display_image()
            else:
                print("Error: Could not read video file")
            
    def display_image(self):
        if self.image is not None:
            height, width = self.image.shape[:2]
            pil_image = Image.fromarray(self.image)
            
            # Resize window to fit image
            self.root.geometry(f"{width}x{height}")
            
            self.photo = ImageTk.PhotoImage(image=pil_image)
            
            # Update canvas
            self.canvas.config(width=width, height=height)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            
    def on_click(self, event):
        # Clear previous point if it exists
        self.clear_point()
        
        # Draw new point
        x, y = event.x, event.y
        self.point_coordinates = (x, y)
        
        # Draw point marker (cross)
        size = self.marker_size
        self.point_marker = [
            self.canvas.create_line(x-size, y, x+size, y, fill='red', width=2),
            self.canvas.create_line(x, y-size, x, y+size, fill='red', width=2)
        ]
        
        print(f"Selected point: ({x}, {y})")
        
    def clear_point(self):
        if self.point_marker:
            for marker in self.point_marker:
                self.canvas.delete(marker)
            self.point_marker = None
            self.point_coordinates = None
            
    def save_coordinates(self):
        if self.point_coordinates:
            x, y = self.point_coordinates
            
            default_filename = "point_image.txt" if self.input_type.get() == "Image" else "point_video.txt"
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt")],
                initialfile=default_filename,
                title="Save Point Coordinates"
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(f"{x},{y}\n")
                
                print(f"Saved coordinates to {file_path}")
                print(f"Point coordinates (x,y): {x},{y}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PointSelector(root)
    root.mainloop()
