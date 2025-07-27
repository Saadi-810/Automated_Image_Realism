import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import os

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Detector")
        self.root.geometry("800x800")  # Increased size to show both original and resized
        self.root.configure(bg="#f0f0f0")
        
        try:
            self.model = tf.keras.models.load_model('best_model.h5')
        except Exception as e:
            messagebox.showerror("Error", f"Could not load model: {str(e)}")
            self.root.destroy()
            return
            
        self.create_widgets()
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="AI Image Detector",
            font=("Helvetica", 24, "bold"),
            padding=(0, 0, 0, 20)
        )
        title_label.pack()
        
        # Instructions
        instructions = ttk.Label(
            main_frame,
            text="Upload an image to determine if it's AI-generated or real.",
            font=("Helvetica", 12),
            wraplength=500,
            padding=(0, 0, 0, 20)
        )
        instructions.pack()
        
        # Create frame for both images
        self.images_frame = ttk.Frame(main_frame)
        self.images_frame.pack(fill=tk.BOTH, expand=True)
        
        # Original image frame
        self.original_frame = ttk.LabelFrame(self.images_frame, text="Original Image", padding="10")
        self.original_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        
        # Resized image frame
        self.resized_frame = ttk.LabelFrame(self.images_frame, text="Processed Image (32x32)", padding="10")
        self.resized_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        
        # Image labels
        self.original_label = ttk.Label(self.original_frame)
        self.original_label.pack(pady=10)
        
        self.resized_label = ttk.Label(self.resized_frame)
        self.resized_label.pack(pady=10)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        # Select button
        self.select_button = ttk.Button(
            button_frame,
            text="Select Image",
            command=self.select_and_classify_image,
            style="Accent.TButton"
        )
        self.select_button.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        self.clear_button = ttk.Button(
            button_frame,
            text="Clear",
            command=self.clear_display,
            state=tk.DISABLED
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            main_frame,
            orient=tk.HORIZONTAL,
            length=400,
            mode='indeterminate'
        )
        self.progress.pack(pady=10)
        
        # Results frame
        self.results_frame = ttk.Frame(main_frame, padding="10")
        self.results_frame.pack(fill=tk.X, expand=True)
        
        # Result label
        self.result_label = ttk.Label(
            self.results_frame,
            text="",
            font=("Helvetica", 12),
            wraplength=500,
            justify=tk.CENTER
        )
        self.result_label.pack()
        
        # Confidence meter
        self.confidence_canvas = tk.Canvas(
            self.results_frame,
            width=400,
            height=40,
            bg="white",
            highlightthickness=0
        )
        self.confidence_canvas.pack(pady=10)

    def preprocess_image(self, image_path, target_size=(32, 32)):
        try:
            # Load original image
            original_img = Image.open(image_path)
            
            # Convert to RGB if necessary (handles PNG with alpha channel)
            if original_img.mode != 'RGB':
                original_img = original_img.convert('RGB')
            
            # Resize image to 32x32 using LANCZOS resampling for better quality
            resized_img = original_img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(resized_img)
            img_array = img_array.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Store both original and resized images for display
            return img_array, original_img, resized_img
        except Exception as e:
            messagebox.showerror("Error", f"Error processing image: {str(e)}")
            return None, None, None

    def classify_image(self, image_array):
        try:
            prediction = self.model.predict(image_array)
            return float(prediction[0][0])
        except Exception as e:
            messagebox.showerror("Error", f"Error during classification: {str(e)}")
            return None

    def update_confidence_meter(self, score):
        self.confidence_canvas.delete("all")
        
        # Draw background
        self.confidence_canvas.create_rectangle(
            0, 0, 400, 40,
            fill="#f0f0f0",
            outline=""
        )
        
        # Draw meter
        meter_width = score * 400
        if score < 0.5:
            color = f"#{int(255 * (1 - score * 2)):02x}0000"  # Red gradient
        else:
            color = f"#00{int(255 * (score * 2 - 1)):02x}00"  # Green gradient
            
        self.confidence_canvas.create_rectangle(
            0, 0, meter_width, 40,
            fill=color,
            outline=""
        )
        
        # Add marker lines
        for i in range(11):
            x = i * 40
            self.confidence_canvas.create_line(
                x, 0, x, 40,
                fill="#666666",
                width=1
            )
            self.confidence_canvas.create_text(
                x, 20,
                text=str(i * 10),
                font=("Helvetica", 8)
            )

    def select_and_classify_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        try:
            # Start progress bar
            self.progress.start()
            self.select_button.configure(state=tk.DISABLED)
            
            # Process the image
            img_array, original_img, resized_img = self.preprocess_image(file_path)
            
            if img_array is not None:
                # Display original image
                aspect_ratio = original_img.width / original_img.height
                new_width = min(300, original_img.width)
                new_height = int(new_width / aspect_ratio)
                display_original = original_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                original_tk = ImageTk.PhotoImage(display_original)
                self.original_label.configure(image=original_tk)
                self.original_label.image = original_tk
                
                # Display resized image (scaled up for visibility)
                display_resized = resized_img.resize((200, 200), Image.Resampling.NEAREST)
                resized_tk = ImageTk.PhotoImage(display_resized)
                self.resized_label.configure(image=resized_tk)
                self.resized_label.image = resized_tk
                
                # Classify the image
                rating = self.classify_image(img_array)
                
                if rating is not None:
                    # Update confidence meter
                    self.update_confidence_meter(rating)
                    
                    # Update result text
                    if rating < 0.3:
                        verdict = "Likely AI-Generated"
                    elif rating < 0.7:
                        verdict = "Uncertain"
                    else:
                        verdict = "Likely Real"
                    
                    self.result_label.configure(
                        text=f"Classification Result: {verdict}\n"
                             f"Confidence Score: {rating:.2%}"
                    )
                
                self.clear_button.configure(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            
        finally:
            # Stop progress bar
            self.progress.stop()
            self.select_button.configure(state=tk.NORMAL)

    def clear_display(self):
        self.original_label.configure(image="")
        self.resized_label.configure(image="")
        self.result_label.configure(text="")
        self.confidence_canvas.delete("all")
        self.clear_button.configure(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.configure("Accent.TButton", font=("Helvetica", 12))
    app = ImageClassifierApp(root)
    root.mainloop()