#!/usr/bin/env python3
"""
CT Bone Segmentation Tool
Processes DICOM CT scans, segments bones based on Hounsfield unit ranges,
and exports the result as a 3D STL file.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pydicom
from scipy import ndimage
from skimage import measure, morphology
import vtk
from stl import mesh
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading


class CTBoneSegmentation:
    def __init__(self, root):
        self.root = root
        self.root.title("CT Bone Segmentation Tool")
        self.root.geometry("1200x800")
        
        # Data storage
        self.dicom_files = []
        self.volume_data = None
        self.segmented_data = None
        self.hu_min = 150  # Default bone HU threshold
        self.hu_max = 3000
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the graphical user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="DICOM File Selection", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(file_frame, text="Select DICOM Folder", 
                  command=self.select_dicom_folder).grid(row=0, column=0, padx=(0, 10))
        
        self.file_label = ttk.Label(file_frame, text="No DICOM files selected")
        self.file_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # HU range selection frame
        hu_frame = ttk.LabelFrame(main_frame, text="Hounsfield Unit Range", padding="5")
        hu_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(hu_frame, text="Min HU:").grid(row=0, column=0, padx=(0, 5))
        self.hu_min_var = tk.StringVar(value=str(self.hu_min))
        ttk.Entry(hu_frame, textvariable=self.hu_min_var, width=10).grid(row=0, column=1, padx=(0, 20))
        
        ttk.Label(hu_frame, text="Max HU:").grid(row=0, column=2, padx=(0, 5))
        self.hu_max_var = tk.StringVar(value=str(self.hu_max))
        ttk.Entry(hu_frame, textvariable=self.hu_max_var, width=10).grid(row=0, column=3, padx=(0, 20))
        
        ttk.Button(hu_frame, text="Preview Segmentation", 
                  command=self.preview_segmentation).grid(row=0, column=4, padx=(10, 0))
        
        # Processing frame
        process_frame = ttk.LabelFrame(main_frame, text="Processing", padding="5")
        process_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(process_frame, text="Load DICOM Series", 
                  command=self.load_dicom_series).grid(row=0, column=0, padx=(0, 10))
        
        ttk.Button(process_frame, text="Segment Bones", 
                  command=self.segment_bones).grid(row=0, column=1, padx=(0, 10))
        
        ttk.Button(process_frame, text="Export STL", 
                  command=self.export_stl).grid(row=0, column=2, padx=(0, 10))
        
        # Progress bar
        self.progress = ttk.Progressbar(process_frame, mode='indeterminate')
        self.progress.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Visualization frame
        viz_frame = ttk.LabelFrame(main_frame, text="Visualization", padding="5")
        viz_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for visualization
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        
    def select_dicom_folder(self):
        """Select folder containing DICOM files"""
        folder_path = filedialog.askdirectory(title="Select DICOM Folder")
        if folder_path:
            self.dicom_files = self.find_dicom_files(folder_path)
            if self.dicom_files:
                self.file_label.config(text=f"Found {len(self.dicom_files)} DICOM files")
                messagebox.showinfo("Success", f"Found {len(self.dicom_files)} DICOM files")
            else:
                self.file_label.config(text="No DICOM files found")
                messagebox.showwarning("Warning", "No DICOM files found in selected folder")
    
    def find_dicom_files(self, folder_path):
        """Find all DICOM files in the specified folder"""
        dicom_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.dcm', '.dicom')):
                    dicom_files.append(os.path.join(root, file))
        return sorted(dicom_files)
    
    def load_dicom_series(self):
        """Load DICOM series and create 3D volume"""
        if not self.dicom_files:
            messagebox.showerror("Error", "Please select DICOM files first")
            return
        
        def load_thread():
            try:
                self.progress.start()
                self.volume_data = self.load_dicom_volume(self.dicom_files)
                self.progress.stop()
                messagebox.showinfo("Success", "DICOM series loaded successfully")
            except Exception as e:
                self.progress.stop()
                messagebox.showerror("Error", f"Failed to load DICOM series: {str(e)}")
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    def load_dicom_volume(self, dicom_files):
        """Load DICOM files and create 3D volume"""
        # Read first file to get dimensions
        first_ds = pydicom.dcmread(dicom_files[0])
        
        # Get image dimensions
        rows = first_ds.Rows
        cols = first_ds.Columns
        num_slices = len(dicom_files)
        
        # Create volume array
        volume = np.zeros((num_slices, rows, cols), dtype=np.float32)
        
        # Load each slice
        for i, file_path in enumerate(dicom_files):
            ds = pydicom.dcmread(file_path)
            if hasattr(ds, 'pixel_array'):
                # Convert to Hounsfield units if needed
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    pixel_array = ds.pixel_array.astype(np.float32)
                    pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
                else:
                    pixel_array = ds.pixel_array.astype(np.float32)
                
                volume[i] = pixel_array
        
        return volume
    
    def preview_segmentation(self):
        """Preview segmentation on middle slice"""
        if self.volume_data is None:
            messagebox.showerror("Error", "Please load DICOM series first")
            return
        
        try:
            self.hu_min = int(self.hu_min_var.get())
            self.hu_max = int(self.hu_max_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid HU values")
            return
        
        # Get middle slice
        middle_slice = self.volume_data.shape[0] // 2
        slice_data = self.volume_data[middle_slice]
        
        # Create segmentation
        bone_mask = (slice_data >= self.hu_min) & (slice_data <= self.hu_max)
        
        # Display results
        self.fig.clear()
        
        # Original slice
        ax1 = self.fig.add_subplot(1, 3, 1)
        ax1.imshow(slice_data, cmap='gray')
        ax1.set_title('Original CT Slice')
        ax1.axis('off')
        
        # Segmentation mask
        ax2 = self.fig.add_subplot(1, 3, 2)
        ax2.imshow(bone_mask, cmap='gray')
        ax2.set_title('Bone Segmentation')
        ax2.axis('off')
        
        # Overlay
        ax3 = self.fig.add_subplot(1, 3, 3)
        ax3.imshow(slice_data, cmap='gray', alpha=0.7)
        ax3.imshow(bone_mask, cmap='Reds', alpha=0.3)
        ax3.set_title('Overlay')
        ax3.axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def segment_bones(self):
        """Segment bones from the 3D volume"""
        if self.volume_data is None:
            messagebox.showerror("Error", "Please load DICOM series first")
            return
        
        try:
            self.hu_min = int(self.hu_min_var.get())
            self.hu_max = int(self.hu_max_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid HU values")
            return
        
        def segment_thread():
            try:
                self.progress.start()
                self.segmented_data = self.create_bone_segmentation(self.volume_data)
                self.progress.stop()
                messagebox.showinfo("Success", "Bone segmentation completed")
            except Exception as e:
                self.progress.stop()
                messagebox.showerror("Error", f"Segmentation failed: {str(e)}")
        
        threading.Thread(target=segment_thread, daemon=True).start()
    
    def create_bone_segmentation(self, volume):
        """Create bone segmentation from 3D volume"""
        # Create binary mask for bone range
        bone_mask = (volume >= self.hu_min) & (volume <= self.hu_max)
        
        # Apply morphological operations to clean up the mask
        # Remove small objects
        bone_mask = morphology.remove_small_objects(bone_mask, min_size=1000)
        
        # Fill holes
        bone_mask = ndimage.binary_fill_holes(bone_mask)
        
        # Apply closing to connect nearby bone structures
        bone_mask = morphology.binary_closing(bone_mask, morphology.ball(2))
        
        return bone_mask.astype(np.uint8)
    
    def export_stl(self):
        """Export segmented bones as STL file"""
        if self.segmented_data is None:
            messagebox.showerror("Error", "Please segment bones first")
            return
        
        def export_thread():
            try:
                self.progress.start()
                
                # Generate mesh using marching cubes
                verts, faces, _, _ = measure.marching_cubes(
                    self.segmented_data, 
                    level=0.5,
                    spacing=(1.0, 1.0, 1.0)
                )
                
                # Create STL mesh
                stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
                for i, face in enumerate(faces):
                    for j in range(3):
                        stl_mesh.vectors[i][j] = verts[face[j], :]
                
                # Save STL file
                output_path = filedialog.asksaveasfilename(
                    defaultextension=".stl",
                    filetypes=[("STL files", "*.stl"), ("All files", "*.*")]
                )
                
                if output_path:
                    stl_mesh.save(output_path)
                    self.progress.stop()
                    messagebox.showinfo("Success", f"STL file saved to {output_path}")
                else:
                    self.progress.stop()
                    
            except Exception as e:
                self.progress.stop()
                messagebox.showerror("Error", f"STL export failed: {str(e)}")
        
        threading.Thread(target=export_thread, daemon=True).start()


def main():
    """Main function"""
    root = tk.Tk()
    app = CTBoneSegmentation(root)
    root.mainloop()


if __name__ == "__main__":
    main()



