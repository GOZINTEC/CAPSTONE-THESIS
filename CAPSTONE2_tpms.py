"""notes: current app is using cell count to control porosity, which is an extensive variable (ata), meaning the porosity changes if the cropped region is bigger/smaller (?).
cell density may be a more reliable setting in the future.

investigate if may distortions sa TPMS patterns.
consider integrating metrics to automatically measure porosity.
possible AI integration: automated FEA para magbigay ng recommended optimal settings for TPMS type (shell/skeletal), thickness or level set, for each pattern type, to maximize strength.
cleaned segmentation is blank for some slices sometimes, even if the output segmentation is correct. investigate.

for printing: recognize that not all generated patterns can be practically printed, particularly through fused deposition modelling. selective laser sintering is more ideal."""
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage import measure, morphology
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import pyvista as pv
import trimesh
import time
import multiprocessing


def pv_viewer_process(verts, faces_flat, color_rgb, window_size=(1000, 800), title="3D Preview"):
    """
    This runs in a separate process. It creates a PyVista Plotter, adds the mesh, and shows it.
    faces_flat must be an int64 1D array in the PolyData format (n, [3, i,j,k] flattened)
    In this script we use the faces format already prepared by np.hstack([3, tri_faces]).
    """
    try:
        # Create PolyData
        pv_mesh = pv.PolyData(verts, faces_flat)
        # Ensure cleaned geometry
        pv_mesh = pv_mesh.clean()

        plotter = pv.Plotter(window_size=window_size, title=title)
        plotter.add_mesh(pv_mesh, color=color_rgb, smooth_shading=True)
        plotter.add_axes()
        # block until user closes
        plotter.show()
    except Exception as e:
        # Print to stderr of the child; main process can't catch easily.
        print("pv_viewer_process error:", e, file=sys.stderr)


def count_non_manifold_edges(mesh):
    try:
        efc = mesh.edges_face_count
        return int(np.sum(efc != 2))
    except Exception:
        return -1


class ProgressDialog:
    """A simple progress dialog that can be updated from a background thread via root.after."""

    def __init__(self, parent, title="Progress", maximum=100):
        self.parent = parent
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.geometry("400x100")
        self.top.transient(parent)
        self.top.grab_set()

        ttk.Label(self.top, text=title).pack(pady=(8, 0))
        self.var = tk.DoubleVar(value=0.0)
        self.bar = ttk.Progressbar(self.top, maximum=maximum, variable=self.var, length=360, mode="determinate")
        self.bar.pack(pady=8)
        self.status = ttk.Label(self.top, text="")
        self.status.pack()

        # prevent the user from closing it directly
        self.top.protocol("WM_DELETE_WINDOW", lambda: None)

    def update(self, value=None, text=None):
        if value is not None:
            # schedule on main thread
            self.parent.after(0, lambda: self.var.set(value))
        if text is not None:
            self.parent.after(0, lambda: self.status.config(text=text))

    def close(self):
        try:
            self.parent.after(0, self.top.destroy)
        except Exception:
            pass


class CTBoneSegmentation:
    def __init__(self, root):
        self.root = root
        self.root.title("CT Bone Segmentation Tool")
        self.root.geometry("1200x800")

        # Data holders
        self.volume_data = None         # numpy array (Z, Y, X)
        self.segmented_data = None      # cleaned 3D binary mask (full volume sized)
        self.image_spacing = (1.0, 1.0, 1.0)
        self.hu_min = 150
        self.hu_max = 3000

        # Crop (Z) defaults - will be set after load
        self.crop_lower = 0
        self.crop_upper = 0

        # preview process handle
        self.preview_process = None
        self.preview_verts = None
        self.preview_faces_flat = None
        self.preview_color = [0.88, 0.84, 0.75]  # bone RGB

        # TPMS pattern controls (defaults)
        self.tpms_pattern = tk.StringVar(value="None")
        self.tpms_invert = tk.BooleanVar(value=False)
        
        self.tpms_cells = tk.IntVar(value=5)       # Represents number of unit cells.
        

        # Add a variable to switch between Skeletal (solid) and Shell (hollow)
        self.tpms_type = tk.StringVar(value="Skeletal") 
        
        # Add a variable for the level-set (iso-value 'c' in the reference)
        self.tpms_level = tk.DoubleVar(value=0.0)      # slider: -2.0 to 2.0
        
        # Add a variable for the wall thickness for Shell TPMS
        self.tpms_thickness = tk.DoubleVar(value=0.1)  # slider: 0.01 to 0.5

        # build GUI
        self.setup_gui()

    # ---------- GUI ----------
    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)

        # file selection
        file_frame = ttk.LabelFrame(main_frame, text="Load CT Data")
        file_frame.grid(row=0, column=0, sticky="ew", pady=4)
        ttk.Button(file_frame, text="Select DICOM Folder", command=self.select_dicom_folder).grid(row=0, column=0, padx=4)
        ttk.Button(file_frame, text="Select NRRD File", command=self.select_nrrd_file).grid(row=0, column=1, padx=4)
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.grid(row=0, column=2, padx=6)

        # HU controls
        hu_frame = ttk.LabelFrame(main_frame, text="HU Threshold")
        hu_frame.grid(row=1, column=0, sticky="ew", pady=4)
        ttk.Label(hu_frame, text="Min HU:").grid(row=0, column=0, padx=(4,2))
        self.hu_min_var = tk.StringVar(value=str(self.hu_min))
        ttk.Entry(hu_frame, textvariable=self.hu_min_var, width=8).grid(row=0, column=1, padx=4)
        ttk.Label(hu_frame, text="Max HU:").grid(row=0, column=2, padx=(8,2))
        self.hu_max_var = tk.StringVar(value=str(self.hu_max))
        ttk.Entry(hu_frame, textvariable=self.hu_max_var, width=8).grid(row=0, column=3, padx=4)
        ttk.Button(hu_frame, text="Preview Segmentation (3D)", command=self.preview_segmentation).grid(row=0, column=4, padx=8)

        ### --- MODIFIED TPMS Scaffold Pattern controls (replaces old frame) --- ###
        pattern_frame = ttk.LabelFrame(main_frame, text="Scaffold Pattern")
        pattern_frame.grid(row=1, column=1, sticky="nsew", pady=4, padx=(8,0))
        
        # Row 0: Pattern Name
        ttk.Label(pattern_frame, text="Pattern:").grid(row=0, column=0, padx=(6,2), sticky="w")
        
        new_patterns = [
            "None", "Gyroid", "Primitive", "Diamond", 
            "Lidinoid", "Split-P", "Schwarz-P"
        ]
        self.pattern_combo = ttk.Combobox(pattern_frame, textvariable=self.tpms_pattern,
                                          values=new_patterns,
                                          state="readonly", width=12)
        self.pattern_combo.grid(row=0, column=1, padx=4, sticky="ew")
        self.pattern_combo.set("Gyroid") # Set a default

        # Row 0: Invert Checkbox
        self.invert_check = ttk.Checkbutton(pattern_frame, text="Invert Pattern", variable=self.tpms_invert)
        self.invert_check.grid(row=0, column=2, columnspan=3, padx=6, sticky="w")

        ### --- MODIFIED (Row 1: Replaces Frequency) --- ###
        # Row 1: Cell Count
        ttk.Label(pattern_frame, text="Cell Count:").grid(row=1, column=0, padx=(6,2), sticky="w")
        self.cells_slider = ttk.Scale(pattern_frame, from_=1, to=100, orient="horizontal",
                                     variable=self.tpms_cells, length=160, command=self._on_cells_change)
        self.cells_slider.grid(row=1, column=1, columnspan=3, padx=4, sticky="ew")
        self.cells_label = ttk.Label(pattern_frame, text=f"{self.tpms_cells.get()}")
        self.cells_label.grid(row=1, column=4, padx=(4,6), sticky="w")
        ### --- END MODIFIED --- ###

        # Row 2: Type (Skeletal/Shell)
        ttk.Label(pattern_frame, text="Type:").grid(row=2, column=0, padx=(6,2), sticky="w")
        self.type_combo = ttk.Combobox(pattern_frame, textvariable=self.tpms_type,
                                         values=["Skeletal", "Shell"],
                                         state="readonly", width=12)
        self.type_combo.grid(row=2, column=1, padx=4, sticky="ew")
        self.type_combo.bind("<<ComboboxSelected>>", self._on_type_change)

        # Row 3: Level-set (for Skeletal)
        self.level_label = ttk.Label(pattern_frame, text="Level-Set:")
        self.level_label.grid(row=3, column=0, padx=(6,2), sticky="w")
        self.level_slider = ttk.Scale(pattern_frame, from_=-2.0, to=2.0, orient="horizontal",
                                      variable=self.tpms_level, length=160, command=self._on_level_change)
        self.level_slider.grid(row=3, column=1, columnspan=3, padx=4, sticky="ew")
        self.level_val_label = ttk.Label(pattern_frame, text=f"{self.tpms_level.get():.2f}")
        self.level_val_label.grid(row=3, column=4, padx=(4,6), sticky="w")

        # Row 4: Thickness (for Shell)
        self.thick_label = ttk.Label(pattern_frame, text="Thickness:")
        self.thick_label.grid(row=4, column=0, padx=(6,2), sticky="w")
        self.thick_slider = ttk.Scale(pattern_frame, from_=0.01, to=0.5, orient="horizontal",
                                      variable=self.tpms_thickness, length=160, command=self._on_thick_change)
        self.thick_slider.grid(row=4, column=1, columnspan=3, padx=4, sticky="ew")
        self.thick_val_label = ttk.Label(pattern_frame, text=f"{self.tpms_thickness.get():.2f}")
        self.thick_val_label.grid(row=4, column=4, padx=(4,6), sticky="w")
        
        # Set initial widget state
        self._on_type_change(None) 

        # actions
        action_frame = ttk.LabelFrame(main_frame, text="Processing")
        action_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(action_frame, text="Load Volume", command=self.load_dicom_series).grid(row=0, column=0, padx=4)
        ttk.Button(action_frame, text="Segment Bones", command=self.segment_bones).grid(row=0, column=1, padx=4)
        ttk.Button(action_frame, text="Smooth Mesh (preview)", command=self.smooth_preview_mesh).grid(row=0, column=2, padx=4)
        ttk.Button(action_frame, text="Export STL", command=self.export_stl).grid(row=0, column=3, padx=4)

        # progress small indicator (indeterminate while long ops occur)
        self.global_progress = ttk.Progressbar(action_frame, mode="indeterminate")
        self.global_progress.grid(row=1, column=0, columnspan=4, sticky="ew", pady=6)

        # slice preview + slider
        viz_frame = ttk.LabelFrame(main_frame, text="Slice Preview & Cropping")
        viz_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")
        viz_frame.rowconfigure(0, weight=1)
        viz_frame.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(10,6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=3, sticky="nsew")

        # Cropping controls (new): lower/upper Z sliders + readouts
        crop_frame = ttk.Frame(viz_frame)
        crop_frame.grid(row=1, column=0, sticky="ew", pady=(8,2))
        crop_frame.columnconfigure(1, weight=1)
        ttk.Label(crop_frame, text="Crop Z Range:").grid(row=0, column=0, padx=4)

        self.crop_lower_var = tk.IntVar(value=0)
        self.crop_upper_var = tk.IntVar(value=0)

        self.crop_lower_slider = ttk.Scale(crop_frame, from_=0, to=0, orient="horizontal", command=self.on_crop_lower_move)
        self.crop_lower_slider.grid(row=0, column=1, sticky="ew", padx=6)
        self.crop_upper_slider = ttk.Scale(crop_frame, from_=0, to=0, orient="horizontal", command=self.on_crop_upper_move)
        self.crop_upper_slider.grid(row=0, column=2, sticky="ew", padx=6)

        # numeric readouts
        self.crop_lower_label = ttk.Label(crop_frame, text="Lower: 0")
        self.crop_lower_label.grid(row=0, column=3, padx=(8,2))
        self.crop_upper_label = ttk.Label(crop_frame, text="Upper: 0")
        self.crop_upper_label.grid(row=0, column=4, padx=(2,8))

        # slice navigation slider (existing)
        self.slice_slider = ttk.Scale(viz_frame, from_=0, to=0, orient="horizontal", command=self.on_slice_slider_move)
        self.slice_slider.grid(row=2, column=0, sticky="ew", pady=6)

        # track selected files
        self.dicom_folder = None
        self.nrrd_file = None

    ###helper methods###
    def _on_cells_change(self, val):
        try:
            # Update label with integer value
            self.cells_label.config(text=f"{int(float(val))}")
        except Exception:
            pass

    def _on_level_change(self, val):
        try:
            self.level_val_label.config(text=f"{float(val):.2f}")
        except Exception:
            pass

    def _on_thick_change(self, val):
        try:
            self.thick_val_label.config(text=f"{float(val):.2f}")
        except Exception:
            pass

    def _on_type_change(self, event):
        """Enable/disable sliders based on TPMS type."""
        try:
            if self.tpms_type.get() == "Skeletal":
                # Enable Level-set, disable Thickness
                self.level_label.config(state="normal")
                self.level_slider.config(state="normal")
                self.level_val_label.config(state="normal")
                self.thick_label.config(state="disabled")
                self.thick_slider.config(state="disabled")
                self.thick_val_label.config(state="disabled")
            else: # "Shell"
                # Disable Level-set, enable Thickness
                self.level_label.config(state="disabled")
                self.level_slider.config(state="disabled")
                self.level_val_label.config(state="disabled")
                self.thick_label.config(state="normal")
                self.thick_slider.config(state="normal")
                self.thick_val_label.config(state="normal")
        except Exception as e:
            print(f"Error in _on_type_change: {e}") # Handle error if widgets not ready

    # ---------- File loading ----------
    def select_dicom_folder(self):
        folder = filedialog.askdirectory(title="Select DICOM Folder")
        if folder:
            self.dicom_folder = folder
            self.file_label.config(text=os.path.basename(folder))

    def select_nrrd_file(self):
        fp = filedialog.askopenfilename(title="Select NRRD File", filetypes=[("NRRD", "*.nrrd"), ("All files", "*.*")])
        if fp:
            self.nrrd_file = fp
            self.file_label.config(text=os.path.basename(fp))

    def load_dicom_series(self):
        """Loads DICOM series or NRRD using SimpleITK. Does not auto-segment."""
        if not getattr(self, "nrrd_file", None) and not getattr(self, "dicom_folder", None):
            messagebox.showwarning("No input", "Select a DICOM folder or NRRD file first")
            return

        def _load():
            try:
                self.global_progress.start()
                if getattr(self, "nrrd_file", None):
                    img = sitk.ReadImage(self.nrrd_file)
                else:
                    reader = sitk.ImageSeriesReader()
                    paths = reader.GetGDCMSeriesFileNames(self.dicom_folder)
                    if not paths:
                        raise RuntimeError("No DICOM series found in folder")
                    reader.SetFileNames(paths)
                    img = reader.Execute()

                arr = sitk.GetArrayFromImage(img).astype(np.int16)  # Z,Y,X
                self.volume_data = arr
                sp = img.GetSpacing()
                # store spacing as (Z,Y,X)
                self.image_spacing = (sp[2], sp[1], sp[0])

                # init sliders
                zmax = max(0, self.volume_data.shape[0]-1)
                # slice slider
                self.slice_slider.config(to=zmax)
                self.slice_slider.set(self.volume_data.shape[0]//2)

                # crop sliders: set full-range defaults
                self.crop_lower = 0
                self.crop_upper = zmax
                self.crop_lower_var.set(self.crop_lower)
                self.crop_upper_var.set(self.crop_upper)
                self.root.after(0, lambda: self.update_crop_sliders_range(zmax))

                # display center slice (threshold preview, not final segmentation)
                self.root.after(0, lambda: self.display_slice(self.volume_data.shape[0]//2))

            except Exception as e:
                messagebox.showerror("Load Error", str(e))
            finally:
                self.global_progress.stop()

        threading.Thread(target=_load, daemon=True).start()

    def update_crop_sliders_range(self, zmax):
        # configure slider ranges and readouts on main thread
        try:
            self.crop_lower_slider.config(to=zmax)
            self.crop_upper_slider.config(to=zmax)
            self.crop_lower_slider.set(self.crop_lower)
            self.crop_upper_slider.set(self.crop_upper)
            self.crop_lower_label.config(text=f"Lower: {self.crop_lower}")
            self.crop_upper_label.config(text=f"Upper: {self.crop_upper}")
        except Exception:
            pass

    # ---------- Slice preview ----------
    def on_slice_slider_move(self, val):
        try:
            idx = int(float(val))
            self.display_slice(idx)
        except Exception:
            pass

    def on_crop_lower_move(self, val):
        """Called while moving the lower crop slider. Enforces lower < upper."""
        try:
            v = int(float(val))
            # ensure at least one slice remains
            if getattr(self, "crop_upper", None) is not None and v >= self.crop_upper:
                # clamp lower to upper-1
                v = max(0, self.crop_upper - 1)
                # update slider position to clamped value
                self.crop_lower_slider.set(v)
            self.crop_lower = v
            self.crop_lower_var.set(v)
            self.crop_lower_label.config(text=f"Lower: {v}")
        except Exception:
            pass

    def on_crop_upper_move(self, val):
        """Called while moving the upper crop slider. Enforces upper > lower."""
        try:
            v = int(float(val))
            if getattr(self, "crop_lower", None) is not None and v <= self.crop_lower:
                v = self.crop_lower + 1
                self.crop_upper_slider.set(v)
            self.crop_upper = v
            self.crop_upper_var.set(v)
            self.crop_upper_label.config(text=f"Upper: {v}")
        except Exception:
            pass

    def display_slice(self, z):
        """Show either threshold preview (before segmentation) or cleaned segmentation slice (if available)."""
        if self.volume_data is None:
            return

        # update HU window from entries
        try:
            self.hu_min = int(self.hu_min_var.get())
            self.hu_max = int(self.hu_max_var.get())
        except Exception:
            pass

        sl = self.volume_data[z]
        threshold_mask = (sl >= self.hu_min) & (sl <= self.hu_max)

        self.fig.clear()
        ax1 = self.fig.add_subplot(1,2,1)
        ax1.imshow(sl, cmap="gray")
        ax1.set_title(f"Slice {z}")
        ax1.axis("off")

        ax2 = self.fig.add_subplot(1,2,2)
        if self.segmented_data is not None:
            if z < self.segmented_data.shape[0]:
                ax2.imshow(self.segmented_data[z], cmap="gray")
            else:
                ax2.text(0.5,0.5,"Segmentation not available for this slice", ha="center")
            ax2.set_title("Cleaned segmentation")
        else:
            ax2.imshow(threshold_mask, cmap="gray")
            ax2.set_title("Threshold preview")
        ax2.axis("off")

        self.canvas.draw()

    # ---------- Segmentation ----------
    def create_bone_mask_keep_largest(self, vol):
        """Create bone mask, remove small objects, fill holes, closing, and keep only the largest connected component."""
        m = (vol >= self.hu_min) & (vol <= self.hu_max)
        # remove very small 3D objects
        m = morphology.remove_small_objects(m, min_size=1000)
        # fill holes (3D)
        m = ndimage.binary_fill_holes(m)
        # closing
        m = morphology.binary_closing(m, morphology.ball(2))

        # label connected components and keep only the largest
        labeled, ncomp = ndimage.label(m)
        if ncomp > 1:
            # compute sizes
            counts = np.bincount(labeled.ravel())
            # counts[0] is background
            counts[0] = 0
            largest_label = np.argmax(counts)
            m = (labeled == largest_label)
        return m.astype(np.uint8)

    ### --- MODIFIED apply_tpms_to_crop (replaces old function) --- ###
    def apply_tpms_to_crop(self, mask_crop, pattern_name, tpms_type, cell_count, level_set, thickness, invert=False):
        """
        Apply a TPMS implicit pattern (Skeletal or Shell) to a cropped binary mask.
        
        - mask_crop: 3D numpy array (Z,Y,X) binary (0/1)
        - pattern_name: "Gyroid", "Primitive", "Diamond", "Lidinoid", "Split-P", "Schwarz-P"
        - tpms_type: "Skeletal" or "Shell"
        - cell_count: Number of repeating unit cells to fit in the largest dimension.
        - level_set: iso-value for Skeletal type (e.g., -2.0 to 2.0)
        - thickness: normalized thickness for Shell type (e.g., 0.01 to 0.5)
        - invert: boolean to invert pattern region
        
        Returns: new binary mask_crop with pattern applied
        """
        if pattern_name is None or pattern_name == "None":
            return mask_crop

        # ensure mask is boolean
        mask_bool = (mask_crop > 0)
        if not np.any(mask_bool):
            print("Mask crop is empty, skipping TPMS.")
            return mask_crop # Don't process empty masks

        zdim, ydim, xdim = mask_bool.shape

        if cell_count < 1: # Safety check
            cell_count = 1

        # --- Coordinate Generation (Fixed in last version) ---
        max_dim = max(zdim, ydim, xdim)
        zz = np.linspace(0.0, zdim / max_dim, zdim, endpoint=False)
        yy = np.linspace(0.0, ydim / max_dim, ydim, endpoint=False)
        xx = np.linspace(0.0, xdim / max_dim, xdim, endpoint=False)
        Z, Y, X = np.meshgrid(zz, yy, xx, indexing="ij")
        
        scale = 2.0 * np.pi * float(cell_count)
        Xs = X * scale
        Ys = Y * scale
        Zs = Z * scale
        # --- End of Fixes ---

        # --- Compute implicit field based on pattern ---
        field = None
        if pattern_name == "Gyroid":
            field = (np.cos(Xs) * np.sin(Ys) + 
                     np.cos(Ys) * np.sin(Zs) + 
                     np.cos(Zs) * np.sin(Xs))
        
        elif pattern_name == "Primitive" or pattern_name == "Schwarz-P":
            field = (np.cos(Xs) + np.cos(Ys) + np.cos(Zs))
        
        elif pattern_name == "Diamond":
            field = (
                np.sin(Xs) * np.sin(Ys) * np.sin(Zs)
                + np.sin(Xs) * np.cos(Ys) * np.cos(Zs)
                + np.cos(Xs) * np.sin(Ys) * np.cos(Zs)
                + np.cos(Xs) * np.cos(Ys) * np.sin(Zs)
            )
        
        elif pattern_name == "Lidinoid":
            field = (
                np.sin(2 * Xs) * np.cos(Ys) * np.sin(Zs)
                + np.sin(Xs) * np.sin(2 * Ys) * np.cos(Zs)
                + np.cos(Xs) * np.sin(Ys) * np.sin(2 * Zs)
                - np.cos(2 * Xs) * np.cos(2 * Ys)
                - np.cos(2 * Ys) * np.cos(2 * Zs)
                - np.cos(2 * Zs) * np.cos(2 * Xs) + 0.3
            )
            
        elif pattern_name == "Split-P":
             field = (
                1.1 * (np.sin(2 * Xs) * np.cos(Ys) * np.sin(Zs) + 
                       np.sin(Xs) * np.sin(2 * Ys) * np.cos(Zs) + 
                       np.cos(Xs) * np.sin(Ys) * np.sin(2 * Zs))
                - 0.2 * (np.cos(2 * Xs) * np.cos(2 * Ys) + 
                         np.cos(2 * Ys) * np.cos(2 * Zs) + 
                         np.cos(2 * Zs) * np.cos(2 * Xs))
                - 0.4 * (np.cos(2 * Xs) + np.cos(2 * Ys) + np.cos(2 * Zs))
            )
        else:
            return mask_crop

        # --- Apply logic based on TPMS Type ---
        if tpms_type == "Skeletal":
            # 'Skeletal' uses the raw field and level_set
            # This creates a solid/void porous block
            pattern_mask = (field > level_set)
        
        elif tpms_type == "Shell":
            # --- NEW FIX: NORMALIZE FIELD FOR SHELL ---
            # This makes the 'thickness' parameter intuitive
            
            # We normalize based on the field values *inside* the bone
            # to prevent outliers from skewing the range.
            field_in_mask = field[mask_bool]
            if field_in_mask.size == 0:
                # No overlap between bone and pattern, return empty
                return np.zeros_like(mask_crop, dtype=np.uint8) 
            
            f_min, f_max = np.min(field_in_mask), np.max(field_in_mask)
            
            if (f_max - f_min) > 1e-5: # Avoid divide-by-zero
                # Normalize the entire field based on the masked range [0, 1]
                field = (field - f_min) / (f_max - f_min)
                # Center the field on 0 (now in range approx [-0.5, 0.5])
                field = field - 0.5
            else:
                # Field is flat (constant value), no pattern can be made
                field.fill(0) 

            # 'Shell' uses the normalized field and thickness
            # 'thickness' slider is [0.01, 0.5].
            # Our normalized field is approx [-0.5, 0.5].
            # So, abs(field) is [0, 0.5].
            # Checking abs(field) < thickness is now an intuitive 1-to-1 mapping.
            pattern_mask = (np.abs(field) < thickness)
            
        else:
            pattern_mask = (field > 0) # Fallback

        if invert:
            pattern_mask = ~pattern_mask

        # Apply pattern only inside the segmented bone region (logical AND)
        combined = np.logical_and(mask_bool, pattern_mask)

        return combined.astype(np.uint8)

    def segment_bones(self):
        if self.volume_data is None:
            messagebox.showwarning("No volume", "Load a volume first")
            return

        def _segment():
            try:
                self.global_progress.start()
                # refresh HU min/max
                try:
                    self.hu_min = int(self.hu_min_var.get())
                    self.hu_max = int(self.hu_max_var.get())
                except Exception:
                    pass

                # validate crop indices
                Z = self.volume_data.shape[0]
                lower = int(self.crop_lower)
                upper = int(self.crop_upper)
                if lower < 0: lower = 0
                if upper >= Z: upper = Z - 1
                if lower >= upper:
                    raise RuntimeError("Invalid crop range: lower must be < upper")

                # crop the volume for segmentation
                vol_crop = self.volume_data[lower:upper+1]

                # run segmentation on cropped volume
                mask_crop = self.create_bone_mask_keep_largest(vol_crop)

                pattern = self.tpms_pattern.get()
                invert = self.tpms_invert.get()
                
                cell_count = self.tpms_cells.get() # Renamed from freq
                
                tpms_type = self.tpms_type.get()
                level_set = self.tpms_level.get()
                thickness = self.tpms_thickness.get()

                if pattern is None:
                    pattern = "None"

                if pattern != "None":
                    try:
                        # apply on cropped mask; returns same shape
                        # Pass all new parameters to the updated function
                        mask_crop = self.apply_tpms_to_crop(
                            mask_crop, 
                            pattern_name=pattern, 
                            tpms_type=tpms_type,
                            cell_count=cell_count, # Pass new parameter
                            level_set=level_set, 
                            thickness=thickness, 
                            invert=invert
                        )
                    except Exception as e:
                        # if TPMS fails, proceed with original mask_crop and report
                        print("TPMS application error:", e, file=sys.stderr)
                ### --- END MODIFICATION --- ###

                # paste back into full-sized segmented_data so slice indices remain consistent
                full_mask = np.zeros_like(self.volume_data, dtype=np.uint8)
                full_mask[lower:upper+1] = mask_crop
                self.segmented_data = full_mask

                # update display to show cleaned segmentation on center slice (or within crop)
                center_slice = min(self.volume_data.shape[0]//2, upper)
                self.root.after(0, lambda: self.display_slice(center_slice))
                messagebox.showinfo("Segmentation", f"Segmentation completed (largest component preserved, crop applied). Pattern: {pattern}{' (inverted)' if invert else ''}")
                # also open 3D preview with cleaned segmentation
                self.root.after(0, lambda: self.show_3d_preview(self.segmented_data))
            except Exception as e:
                messagebox.showerror("Segmentation Error", str(e))
            finally:
                self.global_progress.stop()

        threading.Thread(target=_segment, daemon=True).start()

    # ---------- Preview segmentation based on raw threshold (not cleaned) ----------
    def preview_segmentation(self):
        """Build a threshold mask (not cleaned) using current HU and current crop, then preview in 3D."""
        if self.volume_data is None:
            messagebox.showwarning("No volume", "Load a volume first")
            return

        try:
            self.hu_min = int(self.hu_min_var.get())
            self.hu_max = int(self.hu_max_var.get())
        except Exception:
            pass

        # validate crop indices
        Z = self.volume_data.shape[0]
        lower = int(self.crop_lower)
        upper = int(self.crop_upper)
        if lower < 0: lower = 0
        if upper >= Z: upper = Z - 1
        if lower >= upper:
            messagebox.showwarning("Invalid crop", "Crop lower must be less than crop upper.")
            return

        raw_mask = (self.volume_data >= self.hu_min) & (self.volume_data <= self.hu_max)
        # apply crop and paste into full-sized array for consistent indexing
        full_mask = np.zeros_like(self.volume_data, dtype=np.uint8)
        full_mask[lower:upper+1] = raw_mask[lower:upper+1].astype(np.uint8)
        # show preview (will run marching cubes & smoothing)
        self.show_3d_preview(full_mask)

    # ---------- 3D Preview (separate process) ----------
    def _prepare_mesh_from_mask(self, mask):
        """Runs marching_cubes and returns verts and faces_flat suitable for pv.PolyData."""
        verts, faces, _, _ = measure.marching_cubes(mask.astype(float), level=0.5, spacing=self.image_spacing)
        # assemble faces in the pyvista / vtk flattened format (n, 3, i, j, k) flattened
        faces_with_counts = np.hstack([np.full((faces.shape[0],1), 3, dtype=np.int64), faces.astype(np.int64)])
        faces_flat = faces_with_counts.ravel()
        return verts.astype(np.float64), faces_flat.astype(np.int64)

    def _close_preview_if_open(self):
        """Terminate preview process if running."""
        if getattr(self, "preview_process", None) is not None:
            try:
                if self.preview_process.is_alive():
                    # terminate child process
                    self.preview_process.terminate()
                    self.preview_process.join(timeout=2.0)
            except Exception:
                pass
            finally:
                self.preview_process = None
                self.preview_verts = None
                self.preview_faces_flat = None

    def show_3d_preview(self, mask=None, auto_smooth=True):
        """
        Opens a separate PyVista viewer process showing the mesh.
        If a previous preview exists it is closed first.
        """
        if self.volume_data is None:
            messagebox.showwarning("No volume", "Load a volume first")
            return

        if mask is None:
            mask = (self.volume_data >= self.hu_min) & (self.volume_data <= self.hu_max)

        # ensure mask is binary np.uint8
        mask = (mask > 0).astype(np.uint8)
        
        ### --- Check for empty mask BEFORE marching_cubes --- ###
        if not np.any(mask):
            messagebox.showwarning("3D Preview Error", "Cannot generate 3D preview: the resulting mask is empty.")
            return

        try:
            verts, faces_flat = self._prepare_mesh_from_mask(mask)
        except Exception as e:
            messagebox.showerror("3D Preview Error", f"marching_cubes failed: {e}\n\n(This can happen if the mask is empty or too sparse).")
            return

        # create a PyVista mesh in this process to apply smoothing & hole fill using PyVista functions
        pv_mesh = pv.PolyData(verts, faces_flat).clean()

        # Auto smoothing: initial smoothing + taubin
        if auto_smooth:
            try:
                pv_mesh = pv_mesh.smooth(n_iter=15, relaxation_factor=0.05)
            except Exception:
                pass
            try:
                pv_mesh = pv_mesh.smooth_taubin(n_iter=30, pass_band=0.1)
            except Exception:
                pass

        # fill holes using pyvista (area threshold)
        try:
            pv_mesh = pv_mesh.fill_holes(1000.0)
        except Exception:
            pass

        # final clean
        pv_mesh = pv_mesh.clean()

        # Convert to verts/faces_flat to send to child process (so child constructs its own pv.PolyData).
        child_verts = pv_mesh.points.astype(np.float64)
        child_faces_flat = pv_mesh.faces.astype(np.int64) # This includes the fix from np.int66

        # store preview mesh for further operations
        self.current_pv_mesh = pv_mesh

        # attempt trimesh conversion here for check/metrics (not used for hole filling anymore)
        try:
            self.current_tri = trimesh.Trimesh(vertices=pv_mesh.points, faces=pv_mesh.faces.reshape(-1,4)[:,1:], process=False)
        except Exception:
            self.current_tri = None

        # ensure previous preview is closed
        self._close_preview_if_open()

        # spawn a separate process to show the mesh
        try:
            p = multiprocessing.Process(target=pv_viewer_process, args=(child_verts, child_faces_flat, self.preview_color))
            p.daemon = True
            p.start()
            self.preview_process = p
            # keep last verts/faces_flat around
            self.preview_verts = child_verts
            self.preview_faces_flat = child_faces_flat
        except Exception as e:
            messagebox.showerror("Preview Spawn Error", str(e))

    # ---------- Smoothing preview mesh (keeps preview open) ----------
    def smooth_preview_mesh(self):
        if getattr(self, "current_pv_mesh", None) is None:
            messagebox.showwarning("No preview", "Generate a 3D preview first")
            return

        def _smooth():
            try:
                self.global_progress.start()
                pv_mesh = self.current_pv_mesh.copy()
                try:
                    pv_mesh = pv_mesh.smooth(n_iter=15, relaxation_factor=0.05)
                except Exception:
                    pass
                try:
                    pv_mesh = pv_mesh.smooth_taubin(n_iter=30, pass_band=0.1)
                except Exception:
                    pass
                # fill holes
                try:
                    pv_mesh = pv_mesh.fill_holes(1000.0)
                except Exception:
                    pass
                pv_mesh = pv_mesh.clean()
                self.current_pv_mesh = pv_mesh

                # refresh preview window: close old and reopen new
                self._close_preview_if_open()
                child_verts = pv_mesh.points.astype(np.float64)
                child_faces_flat = pv_mesh.faces.astype(np.int64)
                p = multiprocessing.Process(target=pv_viewer_process, args=(child_verts, child_faces_flat, self.preview_color))
                p.daemon = True
                p.start()
                self.preview_process = p
                self.preview_verts = child_verts
                self.preview_faces_flat = child_faces_flat
            except Exception as e:
                messagebox.showerror("Smoothing Error", str(e))
            finally:
                self.global_progress.stop()

        threading.Thread(target=_smooth, daemon=True).start()

    # ---------- Export (with progress dialog). Closes preview first. ----------
    def export_stl(self):
        """Export using the pipeline:
           1) Ensure preview window closed
           2) Use current_pv_mesh if available, otherwise build from segmented_data
           3) Extract surface, triangulate, clean
           4) Smooth (pv.smooth, pv.smooth_taubin)
           5) Fill holes using pv.Mesh.fill_holes(1000.0)
           6) Convert to trimesh and run reparations
           7) Export to .stl
        """
        if self.segmented_data is None and self.current_pv_mesh is None:
            messagebox.showwarning("No mesh", "No mesh available. Run segmentation or preview first.")
            return

        out_path = filedialog.asksaveasfilename(title="Save STL as", defaultextension=".stl", filetypes=[("STL", "*.stl")])
        if not out_path:
            return

        # close preview to maximize resources
        self._close_preview_if_open()

        progress = ProgressDialog(self.root, title="Exporting STL", maximum=100)

        def _export():
            try:
                step = 0
                total_steps = 5
                def setp(pct, txt=None):
                    progress.update(value=pct, text=txt)

                # STEP 1: obtain pv_mesh
                setp(0, "Preparing mesh...")
                if self.current_pv_mesh is not None:
                    pv_mesh = self.current_pv_mesh.copy()
                else:
                    # build from segmented_data
                    if self.segmented_data is None:
                        raise RuntimeError("No segmentation to export.")
                        
                    ### --- Check for empty mask BEFORE marching_cubes --- ###
                    if not np.any(self.segmented_data):
                        raise RuntimeError("Cannot export: the segmented mask is empty.")

                    verts, faces_flat = self._prepare_mesh_from_mask(self.segmented_data)
                    pv_mesh = pv.PolyData(verts, faces_flat)

                time.sleep(0.05)
                setp(10, "Surface extraction & cleaning...")

                # extract surface, triangulate, clean
                try:
                    pv_mesh = pv_mesh.extract_surface().triangulate().clean(tolerance=1e-3)
                except Exception:
                    # fallback to clean only
                    pv_mesh = pv_mesh.clean()

                time.sleep(0.05)
                setp(35, "Initial smoothing...")
                # STEP 2: initial smoothing
                try:
                    pv_mesh = pv_mesh.smooth(n_iter=15, relaxation_factor=0.05)
                except Exception:
                    pass

                time.sleep(0.05)
                setp(55, "Taubin smoothing & hole filling...")
                # STEP 3: taubin + fill holes (PyVista fill_holes)
                try:
                    pv_mesh = pv_mesh.smooth_taubin(n_iter=30, pass_band=0.1)
                except Exception:
                    pass
                try:
                    pv_mesh = pv_mesh.fill_holes(1000.0)
                except Exception:
                    pass
                pv_mesh = pv_mesh.clean()

                time.sleep(0.05)
                setp(75, "Converting to Trimesh & repairing...")

                # STEP 4: convert to trimesh and repair
                try:
                    tri = trimesh.Trimesh(vertices=pv_mesh.points, faces=pv_mesh.faces.reshape(-1,4)[:,1:], process=False)
                except Exception as e:
                    # as a fallback, attempt simpler face creation
                    faces = pv_mesh.faces.reshape(-1,4)[:,1:]
                    tri = trimesh.Trimesh(vertices=pv_mesh.points, faces=faces, process=False)

                # keep largest component in mesh form too
                try:
                    comps = tri.split(only_watertight=False)
                    if len(comps) > 1:
                        tri = max(comps, key=lambda c: len(c.faces))
                except Exception:
                    pass

                # run trimesh repair steps (safe guarded)
                try:
                    trimesh.repair.fix_normals(tri)
                except Exception:
                    pass
                try:
                    trimesh.repair.fill_holes(tri)
                except Exception:
                    pass
                try:
                    trimesh.repair.fix_inversion(tri)
                except Exception:
                    pass
                try:
                    trimesh.repair.broken_faces(tri)
                except Exception:
                    pass

                time.sleep(0.05)
                setp(90, "Saving STL...")
                # STEP 5: export
                tri.export(out_path)

                setp(100, "Done")
                time.sleep(0.1)
                progress.close()
                messagebox.showinfo("Export Complete", f"Saved STL to:\n{out_path}")
            except Exception as e:
                try:
                    progress.close()
                except Exception:
                    pass
                messagebox.showerror("Export Error", str(e))

        threading.Thread(target=_export, daemon=True).start()


def main():
    # Multiprocessing safety on Windows
    try:
        multiprocessing.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

    root = tk.Tk()
    app = CTBoneSegmentation(root)
    root.mainloop()


if __name__ == "__main__":
    main()
