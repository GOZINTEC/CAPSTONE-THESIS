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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import pyvista as pv
import trimesh
import time
import nrrd

def count_non_manifold_edges(mesh):
    try:
        efc = mesh.edges_face_count
        return int(np.sum(efc != 2))
    except Exception:
        try:
            return int(np.sum(mesh.edges_unique_length > 2))
        except Exception:
            # give -1 to indicate "unknown"
            return -1
# *** REVISION END ***

class CTBoneSegmentation:
    def __init__(self, root):
        self.root = root
        self.root.title("CT Bone Segmentation Tool")
        self.root.geometry("1200x800")
        
        # Data storage
        self.dicom_files = []
        self.volume_data = None
        self.segmented_data = None
        self.nrrd_file = None
        self.nrrd_header = None
        self.volume_type = None
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
        file_frame = ttk.LabelFrame(main_frame, text="DICOM/NRRD File Selection", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(file_frame, text="Select DICOM Folder", 
                  command=self.select_dicom_folder).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(file_frame, text="Select NRRD File",
                   command=self.select_nrrd_file).grid(row=0, column=2, padx=(10,10))
        
        self.file_label = ttk.Label(file_frame, text="No DICOM/NRRD files selected")
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
        
        ttk.Button(process_frame, text="Load DICOM/NRRD Series", 
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
            self.nrrd_file = None
            self.nrrd_header = None
            if self.dicom_files:
                self.file_label.config(text=f"Found {len(self.dicom_files)} DICOM files")
                messagebox.showinfo("Success", f"Found {len(self.dicom_files)} DICOM files")
            else:
                self.file_label.config(text="No DICOM files found")
                messagebox.showwarning("Warning", "No DICOM files found in selected folder")
                
    def select_nrrd_file(self):
        file_path = filedialog.askopenfilename(
            title="Select NRRD File",
            filetypes=[("NRRD files", "*.nrrd"), ("All files", "*.*")]
        )
        if file_path:
            self.nrrd_file = file_path
            self.nrrd_header = None
            self.dicom_files = []
            self.file_label.config(text=f"Selected NRRD file: {os.path.basename(file_path)}")
            messagebox.showinfo("Success", f"Selected NRRD file:\n{file_path}")
        else:
            self.nrrd_file = None
        
    def find_dicom_files(self, folder_path):
        """Find all DICOM files in the specified folder"""
        dicom_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.dcm', '.dicom')):
                    dicom_files.append(os.path.join(root, file))
        return sorted(dicom_files)
    
    def load_nrrd_volume(self, nrrd_file):
        """Load an NRRD file and return (volume, header)"""
        data, header = nrrd.read(nrrd_file)
        return data, header

    def load_dicom_series(self):
        """Load DICOM series or NRRD and create 3D volume"""
        if not self.dicom_files and not getattr(self, "nrrd_file", None):
            messagebox.showerror("Error", "Please select a valid DICOM or NRRD file first")
            return
        
        def load_thread():
            try:
                self.progress.start()
                if self.dicom_files:
                    self.volume_data = self.load_dicom_volume(self.dicom_files)
                    self.volume_type = "dicom"
                elif self.nrrd_file:
                    self.volume_data, self.nrrd_header = self.load_nrrd_volume(self.nrrd_file)
                    self.volume_type = "nrrd"
                self.progress.stop()
                messagebox.showinfo("Success", "Volume loaded successfully")

            except Exception as e:
                self.progress.stop()
                messagebox.showerror("Error", f"Failed to load volume: {str(e)}")
        
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
            messagebox.showerror("Error", "Please load DICOM/NRRD series first")
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
            messagebox.showerror("Error", "Please load DICOM/NRRD series first")
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
        """Export segmented bones as STL file with smoothing and mesh fixing"""
        if self.segmented_data is None:
            messagebox.showerror("Error", "Please segment bones first")
            return

        # *** REVISION START ***
        def log_message(msg):
            print(msg)
            try:
                self.file_label.config(text=msg)
                self.root.update_idletasks()
            except Exception:
                pass

        def set_buttons_state(state):
            # This iterates the widget tree and disables/enables ttk.Buttons
            for widget in self.root.winfo_children():
                for child in widget.winfo_children():
                    for subchild in child.winfo_children():
                        if isinstance(subchild, ttk.Button):
                            if state == 'disabled':
                                subchild.state(['disabled'])
                            else:
                                subchild.state(['!disabled'])
        # *** REVISION END ***

        def export_thread():
            try:
                # disable buttons
                self.root.after(0, lambda: set_buttons_state('disabled'))
                self.progress.config(mode='determinate', maximum=100, value=0)
                log_message("Starting STL export...")

                total_steps = 4
                progress_per_step = 100 / total_steps
                current_progress = 0

                # Step 1: Marching cubes
                log_message("STEP 1: running Marching Cubes (extracting 3D surface)...")
                if getattr(self,"volume_type", None)=="nrrd":
                    nrrd_space_dirs = self.nrrd_header.get('space directions', [[1,0,0],[0,1,0],[0,0,1]])
                    spacing = [np.linalg.norm(np.array(nrrd_space_dirs[i])) for i in range(3)]
                elif self.dicom_files:
                    ds = pydicom.dcmread(self.dicom_files[0])
                    ps = getattr(ds, "PixelSpacing", [1.0,1.0])
                    st = getattr(ds, "SliceThickness", 1.0)
                    spacing = [float(st), float(ps[1]), float(ps[0])]
                else:
                    spacing = (1.0,1.0,1.0)
                verts, faces, normals, values = measure.marching_cubes(
                    self.segmented_data,
                    level=0.5,
                    spacing=spacing
                )

                current_progress += progress_per_step
                self.progress['value'] = current_progress
                self.progress.update_idletasks()
                log_message(f"STEP 1 COMPLETE ({int(current_progress)}%)")
                time.sleep(0.05)

                # Step 2: create a PyVista mesh & smooth
                log_message("STEP 2: creating PyVista mesh and cleaning...")
                faces_with_counts = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
                faces_flat = faces_with_counts.ravel()
                pv_mesh = pv.PolyData(verts, faces_flat)
                try:
                    pv_mesh = pv_mesh.extract_surface().triangulate().clean(tolerance=1e-3)
                    log_message("Surface extracted, triangulated, and cleaned")
                except Exception as e:
                    log_message(f"Surface extraction failed: {e}")

                try:
                    pv_mesh = pv_mesh.smooth(n_iter=15, relaxation_factor= 0.05)
                    log_message("Initial smoothing applied")
                except Exception:
                    log_message("skipping initial smoothing due to error")

                current_progress += progress_per_step
                self.progress['value'] = current_progress
                self.progress.update_idletasks()
                log_message("STEP 2 COMPLETE")
                time.sleep(0.05)

                # Step 3: convert to trimesh and repair
                log_message("STEP 3: voxel remeshing, taubin, smoothing, and trimesh repair")
                pv_mesh = pv_mesh.fill_holes(1000.0)  # Adjust area as needed

                try:
                    voxel_surface = pv_mesh.smooth_taubin(n_iter=30, pass_band=.1)
                    log_message("Applied taubin smoothing to final mesh")
                except Exception as e:
                    voxel_surface = pv_mesh
                    log_message(f"taubin smoothing skipped:{e}")

                try:
                    tri_tmp = trimesh.Trimesh(vertices=voxel_surface.points,
                                              faces=voxel_surface.faces.reshape(-1,4)[:,1:],
                                              process=False)
                    components = tri_tmp.split(only_watertight=False)
                    if len(components)>1:
                        tri_tmp=max(components, key=lambda c:len(c.faces))
                        log_message(f"removed {len(components)-1} smaller components")
                    else:
                        log_message("single connected component detected")
                    tri_clean=tri_tmp
                except Exception as e:
                    log_message(f"connected component filtering failed: {e}")
                    tri_clean = tri_tmp

                try:
                    tri=tri_clean
                except Exception as e:
                    log_message(f"falling back to original voxel model without component reduction: {e}")
                    pv_faces = voxel_surface.faces.reshape(-1,4)[:,1:]
                    tri = trimesh.Trimesh(vertices=voxel_surface.points, faces=pv_faces, process=False)

                before_nm = count_non_manifold_edges(tri)
                before_watertight = bool(tri.is_watertight)
                log_message(f"Before repair: {before_nm} non-manifold edges | Watertight: {before_watertight}")

                try:
                    trimesh.repair.fix_normals(tri)
                    trimesh.repair.fill_holes(tri)
                    trimesh.repair.fix_inversion(tri)
                    trimesh.repair.broken_faces(tri)
                except Exception as e:
                    log_message(f"MESH REPAIR SKIPPED DUE TO ERROR: {e}")

                after_nm = count_non_manifold_edges(tri)
                after_watertight = bool(tri.is_watertight)
                log_message(f"After repair: {after_nm} non-manifold edges | Watertight: {after_watertight}")

                current_progress += progress_per_step
                self.progress['value'] = current_progress
                self.progress.update_idletasks()
                log_message(f"STEP 3 COMPLETE ({int(current_progress)}%)")
                time.sleep(0.05)

                # Step 4: save
                log_message("STEP 4: saving STL file...")
                output_path = filedialog.asksaveasfilename(
                    defaultextension=".stl",
                    filetypes=[("STL files", "*.stl"), ("All files", "*.*")],
                    title="Save STL as..."
                )
                if output_path:
                    try:
                        tri.export(output_path, file_type='stl')
                        messagebox.showinfo(
                            "STL Export Complete",
                            "STL saved to:\n{}\n\nBefore repair: {} non-manifold edges\nAfter repair:  {} non-manifold edges\nWatertight: {}".format(
                                output_path, before_nm, after_nm, after_watertight
                            )
                        )
                    except Exception as e:
                        raise

                self.progress.stop()
                self.progress['value'] = 100

            except Exception as e:
                messagebox.showerror("Error", f"STL export failed: {str(e)}")
                log_message(f"❌ Error during export: {e}")

            finally:
                # re-enable UI and reset progress widget to indeterminate mode
                self.root.after(0, lambda: set_buttons_state('!disabled'))
                self.progress.stop()
                try:
                    self.progress.config(mode='indeterminate', value=0)
                except Exception:
                    pass
                self.root.update_idletasks()

        threading.Thread(target=export_thread, daemon=True).start()



def main():
    """Main function"""
    root = tk.Tk()
    app = CTBoneSegmentation(root)
    root.mainloop()


if __name__ == "__main__":
    main()
