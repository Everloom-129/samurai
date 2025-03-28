# create a UI to select a bounding box from an image
import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk

class BoundingBoxSelector:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("Select Bounding Box")
        
        # Initialize variables
        self.image = None
        self.photo = None
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        self.bbox_coordinates = None
        
        # Load and display image
        self.load_and_display_image(image_path)
        
    def load_and_display_image(self, image_path):
        # Load image
        if image_path.endswith(('.mp4', '.avi')):
            cap = cv2.VideoCapture(image_path)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise ValueError("Could not read video file")
            self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            self.image = cv2.imread(image_path)
            if self.image is None:
                raise ValueError("Could not read image file")
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL format and display
        height, width = self.image.shape[:2]
        pil_image = Image.fromarray(self.image)
        self.photo = ImageTk.PhotoImage(image=pil_image)
        
        # Create and configure canvas
        self.canvas = tk.Canvas(self.root, cursor="cross")
        self.canvas.config(width=width, height=height)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
    def on_mouse_down(self, event):
        self.start_x = event.x
        self.start_y = event.y
        
    def on_mouse_move(self, event):
        if self.start_x and self.start_y:
            if self.current_rect:
                self.canvas.delete(self.current_rect)
            self.current_rect = self.canvas.create_rectangle(
                self.start_x, self.start_y, event.x, event.y,
                outline='red', width=2
            )
            
    def on_mouse_up(self, event):
        if self.start_x and self.start_y:
            x1 = min(self.start_x, event.x)
            y1 = min(self.start_y, event.y)
            x2 = max(self.start_x, event.x)
            y2 = max(self.start_y, event.y)
            self.bbox_coordinates = (x1, y1, x2, y2)
            self.root.quit()

def get_bbox_coordinates(image_path):
    root = tk.Tk()
    app = BoundingBoxSelector(root, image_path)
    root.mainloop()
    root.destroy()
    return app.bbox_coordinates

if __name__ == "__main__":
    # Example usage
    image_path = "path/to/your/image_or_video"
    bbox = get_bbox_coordinates(image_path)
    if bbox:
        print(f"Selected bounding box: {bbox}")