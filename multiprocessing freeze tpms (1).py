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
import hashlib
import json
import binascii

# -----------------------------
# Security utilities (simple PBKDF2 password storage)
# -----------------------------
USERS_FILE = os.path.join(os.path.expanduser("~"), ".tpms_users.json")
LOCKOUT_SECONDS = 300  # 5 minutes lockout after repeated failures
MAX_FAILED = 5


def _hash_password(password: str, salt: bytes = None):
    if salt is None:
        salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode('utf-8'), salt, 100_000)
    return binascii.hexlify(salt).decode('ascii'), binascii.hexlify(dk).decode('ascii')


def _verify_password(password: str, salt_hex: str, dk_hex: str):
    salt = binascii.unhexlify(salt_hex.encode('ascii'))
    dk = hashlib.pbkdf2_hmac("sha256", password.encode('utf-8'), salt, 100_000)
    return binascii.hexlify(dk).decode('ascii') == dk_hex


def load_users():
    if not os.path.exists(USERS_FILE):
        create_default_user()
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def save_users(d):
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(d, f, indent=2)
    except Exception as e:
        print("Could not save users:", e)


def create_default_user():
    # default admin/admin
    salt, dk = _hash_password("admin")
    users = {
        "admin": {"salt": salt, "dk": dk, "failed": 0, "locked_until": 0}
    }
    save_users(users)
    return users


def verify_user(username, password):
    users = load_users()
    user = users.get(username)
    now = time.time()
    if not user:
        return False, "User not found"
    if user.get('locked_until', 0) > now:
        return False, f"Account locked until {time.ctime(user.get('locked_until'))}"
    ok = _verify_password(password, user['salt'], user['dk'])
    if ok:
        # reset counters
        user['failed'] = 0
        user['locked_until'] = 0
        save_users(users)
        return True, "OK"
    else:
        user['failed'] = user.get('failed', 0) + 1
        if user['failed'] >= MAX_FAILED:
            user['locked_until'] = now + LOCKOUT_SECONDS
        save_users(users)
        return False, "Incorrect password"

# -----------------------------
# PyVista child process viewer (unchanged)
# -----------------------------

def pv_viewer_process(verts, faces_flat, color_rgb, window_size=(1000, 800), title="3D Preview"):
    try:
        pv_mesh = pv.PolyData(verts, faces_flat)
        pv_mesh = pv_mesh.clean()
        plotter = pv.Plotter(window_size=window_size, title=title)
        plotter.add_mesh(pv_mesh, color=color_rgb, smooth_shading=True)
        plotter.add_axes()
        plotter.show()
    except Exception as e:
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

        # Set background for the Toplevel window to match the theme
        self.top.configure(bg='#087830')
        
        ttk.Label(self.top, text=title).pack(pady=(8, 0))
        self.var = tk.DoubleVar(value=0.0)
        self.bar = ttk.Progressbar(self.top, maximum=maximum, variable=self.var, length=360, mode="determinate")
        self.bar.pack(pady=8)
        self.status = ttk.Label(self.top, text="")
        self.status.pack()

        self.top.protocol("WM_DELETE_WINDOW", lambda: None)

    def update(self, value=None, text=None):
        if value is not None:
            self.parent.after(0, lambda: self.var.set(value))
        if text is not None:
            self.parent.after(0, lambda: self.status.config(text=text))

    def close(self):
        try:
            self.parent.after(0, self.top.destroy)
        except Exception:
            pass


# =============================
# Main Application
# =============================
class CTBoneSegmentation:
    def __init__(self, root):
        self.root = root
        self.root.title("CT Bone Segmentation Tool")

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
        self.tpms_type = tk.StringVar(value="Skeletal")
        self.tpms_level = tk.DoubleVar(value=0.0)
        self.tpms_thickness = tk.DoubleVar(value=0.1)

        # build GUI
        self.setup_gui()

    def setup_gui(self):
        # Common color vars for tk widgets
        THEME_BG = '#087830'
        BTN_BG = '#0a5d2a'
        FG = '#ffffff'

        # top menu with logout
        menubar = tk.Menu(self.root)
        # Configure Menu colors
        account_menu = tk.Menu(menubar, tearoff=0, bg=THEME_BG, fg=FG)
        account_menu.add_command(label="Logout", command=self.logout)
        account_menu.add_command(label="Exit", command=self.exit_app) # Updated command
        menubar.add_cascade(label="Account", menu=account_menu)
        self.root.config(menu=menubar)

        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=0)
        main_frame.rowconfigure(3, weight=1) 

        # file selection
        file_frame = ttk.LabelFrame(main_frame, text="Load CT Data")
        file_frame.grid(row=0, column=0, sticky="ew", pady=4, columnspan=2)
        
        # Use tk.Button for more control over color
        tk.Button(file_frame, text="Select DICOM Folder", command=self.select_dicom_folder, 
                  bg=BTN_BG, fg=FG, activebackground='#0c9239', activeforeground=FG).grid(row=0, column=0, padx=4)
        tk.Button(file_frame, text="Select NRRD File", command=self.select_nrrd_file, 
                  bg=BTN_BG, fg=FG, activebackground='#0c9239', activeforeground=FG).grid(row=0, column=1, padx=4)
        
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
        
        tk.Button(hu_frame, text="Preview Segmentation (3D)", command=self.preview_segmentation, 
                  bg=BTN_BG, fg=FG, activebackground='#0c9239', activeforeground=FG).grid(row=0, column=4, padx=8)

        pattern_frame = ttk.LabelFrame(main_frame, text="Scaffold Pattern")
        pattern_frame.grid(row=1, column=1, sticky="nsew", pady=4, padx=(8,0))
        pattern_frame.columnconfigure(1, weight=1) 
        pattern_frame.columnconfigure(2, weight=1) 
        pattern_frame.columnconfigure(3, weight=1) 
        
        ttk.Label(pattern_frame, text="Pattern:").grid(row=0, column=0, padx=(6,2), sticky="w")

        new_patterns = [
            "None", "Gyroid", "Primitive", "Diamond",
            "Lidinoid", "Split-P", "Schwarz-P"
        ]
        self.pattern_combo = ttk.Combobox(pattern_frame, textvariable=self.tpms_pattern,
                                          values=new_patterns,
                                          state="readonly", width=12)
        self.pattern_combo.grid(row=0, column=1, padx=4, sticky="ew")
        self.pattern_combo.set("Gyroid")

        self.invert_check = ttk.Checkbutton(pattern_frame, text="Invert Pattern", variable=self.tpms_invert)
        self.invert_check.grid(row=0, column=2, columnspan=3, padx=6, sticky="w")

        ttk.Label(pattern_frame, text="Cell Count:").grid(row=1, column=0, padx=(6,2), sticky="w")
        self.cells_slider = ttk.Scale(pattern_frame, from_=1, to=100, orient="horizontal",
                                     variable=self.tpms_cells, length=160, command=self._on_cells_change)
        self.cells_slider.grid(row=1, column=1, columnspan=3, padx=4, sticky="ew")
        self.cells_label = ttk.Label(pattern_frame, text=f"{self.tpms_cells.get()}")
        self.cells_label.grid(row=1, column=4, padx=(4,6), sticky="w")

        ttk.Label(pattern_frame, text="Type:").grid(row=2, column=0, padx=(6,2), sticky="w")
        self.type_combo = ttk.Combobox(pattern_frame, textvariable=self.tpms_type,
                                         values=["Skeletal", "Shell"],
                                         state="readonly", width=12)
        self.type_combo.grid(row=2, column=1, padx=4, sticky="ew")
        self.type_combo.bind("<<ComboboxSelected>>", self._on_type_change)

        self.level_label = ttk.Label(pattern_frame, text="Level-Set:")
        self.level_label.grid(row=3, column=0, padx=(6,2), sticky="w")
        self.level_slider = ttk.Scale(pattern_frame, from_=-2.0, to=2.0, orient="horizontal",
                                      variable=self.tpms_level, length=160, command=self._on_level_change)
        self.level_slider.grid(row=3, column=1, columnspan=3, padx=4, sticky="ew")
        self.level_val_label = ttk.Label(pattern_frame, text=f"{self.tpms_level.get():.2f}")
        self.level_val_label.grid(row=3, column=4, padx=(4,6), sticky="w")

        self.thick_label = ttk.Label(pattern_frame, text="Thickness:")
        self.thick_label.grid(row=4, column=0, padx=(6,2), sticky="w")
        self.thick_slider = ttk.Scale(pattern_frame, from_=0.01, to=0.5, orient="horizontal",
                                      variable=self.tpms_thickness, length=160, command=self._on_thick_change)
        self.thick_slider.grid(row=4, column=1, columnspan=3, padx=4, sticky="ew")
        self.thick_val_label = ttk.Label(pattern_frame, text=f"{self.tpms_thickness.get():.2f}")
        self.thick_val_label.grid(row=4, column=4, padx=(4,6), sticky="w")

        self._on_type_change(None)

        action_frame = ttk.LabelFrame(main_frame, text="Processing")
        action_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=4)
        
        # Use tk.Button for more control over color
        tk.Button(action_frame, text="Load Volume", command=self.load_dicom_series, 
                  bg=BTN_BG, fg=FG, activebackground='#0c9239', activeforeground=FG).grid(row=0, column=0, padx=4)
        tk.Button(action_frame, text="Segment Bones", command=self.segment_bones, 
                  bg=BTN_BG, fg=FG, activebackground='#0c9239', activeforeground=FG).grid(row=0, column=1, padx=4)
        tk.Button(action_frame, text="Smooth Mesh (preview)", command=self.smooth_preview_mesh, 
                  bg=BTN_BG, fg=FG, activebackground='#0c9239', activeforeground=FG).grid(row=0, column=2, padx=4)
        tk.Button(action_frame, text="Export STL", command=self.export_stl, 
                  bg=BTN_BG, fg=FG, activebackground='#0c9239', activeforeground=FG).grid(row=0, column=3, padx=4)

        self.global_progress = ttk.Progressbar(action_frame, mode="indeterminate")
        self.global_progress.grid(row=1, column=0, columnspan=4, sticky="ew", pady=6)

        # Visualization frame
        viz_frame = ttk.LabelFrame(main_frame, text="Slice Preview & Cropping")
        viz_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
        viz_frame.rowconfigure(0, weight=1)
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.columnconfigure(1, weight=1)
        viz_frame.columnconfigure(2, weight=1)

        self.fig = Figure(figsize=(10,6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=3, sticky="nsew")

        crop_frame = ttk.Frame(viz_frame)
        crop_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8,2))
        crop_frame.columnconfigure(1, weight=1)
        crop_frame.columnconfigure(2, weight=1)
        ttk.Label(crop_frame, text="Crop Z Range:").grid(row=0, column=0, padx=4)

        self.crop_lower_var = tk.IntVar(value=0)
        self.crop_upper_var = tk.IntVar(value=0)

        self.crop_lower_slider = ttk.Scale(crop_frame, from_=0, to=0, orient="horizontal", command=self.on_crop_lower_move)
        self.crop_lower_slider.grid(row=0, column=1, sticky="ew", padx=6)
        self.crop_upper_slider = ttk.Scale(crop_frame, from_=0, to=0, orient="horizontal", command=self.on_crop_upper_move)
        self.crop_upper_slider.grid(row=0, column=2, sticky="ew", padx=6)

        self.crop_lower_label = ttk.Label(crop_frame, text="Lower: 0")
        self.crop_lower_label.grid(row=0, column=3, padx=(8,2))
        self.crop_upper_label = ttk.Label(crop_frame, text="Upper: 0")
        self.crop_upper_label.grid(row=0, column=4, padx=(2,8))

        self.slice_slider = ttk.Scale(viz_frame, from_=0, to=0, orient="horizontal", command=self.on_slice_slider_move)
        self.slice_slider.grid(row=2, column=0, columnspan=3, sticky="ew", pady=6)

        self.dicom_folder = None
        self.nrrd_file = None
        
        # Configure matplotlib figure background
        self.fig.patch.set_facecolor(THEME_BG)
        self.fig.patch.set_edgecolor(THEME_BG)

    def _on_cells_change(self, val):
        try:
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
        try:
            if self.tpms_type.get() == "Skeletal":
                self.level_label.config(state="normal")
                self.level_slider.config(state="normal")
                self.level_val_label.config(state="normal")
                self.thick_label.config(state="disabled")
                self.thick_slider.config(state="disabled")
                self.thick_val_label.config(state="disabled")
            else:
                self.level_label.config(state="disabled")
                self.level_slider.config(state="disabled")
                self.level_val_label.config(state="disabled")
                self.thick_label.config(state="normal")
                self.thick_slider.config(state="normal")
                self.thick_val_label.config(state="normal")
        except Exception as e:
            print(f"Error in _on_type_change: {e}")

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

                arr = sitk.GetArrayFromImage(img).astype(np.int16)
                self.volume_data = arr
                sp = img.GetSpacing()
                self.image_spacing = (sp[2], sp[1], sp[0])

                zmax = max(0, self.volume_data.shape[0]-1)
                self.slice_slider.config(to=zmax)
                self.slice_slider.set(self.volume_data.shape[0]//2)

                self.crop_lower = 0
                self.crop_upper = zmax
                self.crop_lower_var.set(self.crop_lower)
                self.crop_upper_var.set(self.crop_upper)
                self.root.after(0, lambda: self.update_crop_sliders_range(zmax))

                self.root.after(0, lambda: self.display_slice(self.volume_data.shape[0]//2))

            except Exception as e:
                messagebox.showerror("Load Error", str(e))
            finally:
                self.global_progress.stop()

        threading.Thread(target=_load, daemon=True).start()

    def update_crop_sliders_range(self, zmax):
        try:
            self.crop_lower_slider.config(to=zmax)
            self.crop_upper_slider.config(to=zmax)
            self.crop_lower_slider.set(self.crop_lower)
            self.crop_upper_slider.set(self.crop_upper)
            self.crop_lower_label.config(text=f"Lower: {self.crop_lower}")
            self.crop_upper_label.config(text=f"Upper: {self.crop_upper}")
        except Exception:
            pass

    def on_slice_slider_move(self, val):
        try:
            idx = int(float(val))
            self.display_slice(idx)
        except Exception:
            pass

    def on_crop_lower_move(self, val):
        try:
            v = int(float(val))
            if getattr(self, "crop_upper", None) is not None and v >= self.crop_upper:
                v = max(0, self.crop_upper - 1)
                self.crop_lower_slider.set(v)
            self.crop_lower = v
            self.crop_lower_var.set(v)
            self.crop_lower_label.config(text=f"Lower: {v}")
        except Exception:
            pass

    def on_crop_upper_move(self, val):
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
        if self.volume_data is None:
            return
        try:
            self.hu_min = int(self.hu_min_var.get())
            self.hu_max = int(self.hu_max_var.get())
        except Exception:
            pass

        sl = self.volume_data[z]
        # Get theme background color for plotting
        THEME_BG = '#087830'
        
        threshold_mask = (sl >= self.hu_min) & (sl <= self.hu_max)

        self.fig.clear()
        ax1 = self.fig.add_subplot(1,2,1)
        ax1.imshow(sl, cmap="gray")
        ax1.set_title(f"Slice {z}", color='#ffffff')
        ax1.axis("off")
        ax1.set_facecolor(THEME_BG) 

        ax2 = self.fig.add_subplot(1,2,2)
        if self.segmented_data is not None:
            if z < self.segmented_data.shape[0]:
                ax2.imshow(self.segmented_data[z], cmap="gray")
            else:
                ax2.text(0.5,0.5,"Segmentation not available for this slice", ha="center", color='#ffffff')
            ax2.set_title("Cleaned segmentation", color='#ffffff')
        else:
            ax2.imshow(threshold_mask, cmap="gray")
            ax2.set_title("Threshold preview", color='#ffffff')
        ax2.axis("off")
        ax2.set_facecolor(THEME_BG)

        self.canvas.draw()

    def create_bone_mask_keep_largest(self, vol):
        m = (vol >= self.hu_min) & (vol <= self.hu_max)
        m = morphology.remove_small_objects(m, min_size=1000)
        m = ndimage.binary_fill_holes(m)
        m = morphology.binary_closing(m, morphology.ball(2))

        labeled, ncomp = ndimage.label(m)
        if ncomp > 1:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            largest_label = np.argmax(counts)
            m = (labeled == largest_label)
        return m.astype(np.uint8)

    def apply_tpms_to_crop(self, mask_crop, pattern_name, tpms_type, cell_count, level_set, thickness, invert=False):
        if pattern_name is None or pattern_name == "None":
            return mask_crop

        mask_bool = (mask_crop > 0)
        if not np.any(mask_bool):
            print("Mask crop is empty, skipping TPMS.")
            return mask_crop

        zdim, ydim, xdim = mask_bool.shape
        if cell_count < 1:
            cell_count = 1

        max_dim = max(zdim, ydim, xdim)
        zz = np.linspace(0.0, zdim / max_dim, zdim, endpoint=False)
        yy = np.linspace(0.0, ydim / max_dim, ydim, endpoint=False)
        xx = np.linspace(0.0, xdim / max_dim, xdim, endpoint=False)
        Z, Y, X = np.meshgrid(zz, yy, xx, indexing="ij")
        scale = 2.0 * np.pi * float(cell_count)
        Xs = X * scale
        Ys = Y * scale
        Zs = Z * scale

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

        if tpms_type == "Skeletal":
            pattern_mask = (field > level_set)
        elif tpms_type == "Shell":
            field_in_mask = field[mask_bool]
            if field_in_mask.size == 0:
                return np.zeros_like(mask_crop, dtype=np.uint8)
            f_min, f_max = np.min(field_in_mask), np.max(field_in_mask)
            if (f_max - f_min) > 1e-5:
                field = (field - f_min) / (f_max - f_min)
                field = field - 0.5
            else:
                field.fill(0)
            pattern_mask = (np.abs(field) < thickness)
        else:
            pattern_mask = (field > 0)

        if invert:
            pattern_mask = ~pattern_mask

        combined = np.logical_and(mask_bool, pattern_mask)
        return combined.astype(np.uint8)

    def segment_bones(self):
        if self.volume_data is None:
            messagebox.showwarning("No volume", "Load a volume first")
            return

        def _segment():
            try:
                self.global_progress.start()
                try:
                    self.hu_min = int(self.hu_min_var.get())
                    self.hu_max = int(self.hu_max_var.get())
                except Exception:
                    pass

                Z = self.volume_data.shape[0]
                lower = int(self.crop_lower)
                upper = int(self.crop_upper)
                if lower < 0: lower = 0
                if upper >= Z: upper = Z - 1
                if lower >= upper:
                    raise RuntimeError("Invalid crop range: lower must be < upper")

                vol_crop = self.volume_data[lower:upper+1]
                mask_crop = self.create_bone_mask_keep_largest(vol_crop)

                pattern = self.tpms_pattern.get()
                invert = self.tpms_invert.get()
                cell_count = self.tpms_cells.get()
                tpms_type = self.tpms_type.get()
                level_set = self.tpms_level.get()
                thickness = self.tpms_thickness.get()

                if pattern is None:
                    pattern = "None"

                if pattern != "None":
                    try:
                        mask_crop = self.apply_tpms_to_crop(
                            mask_crop,
                            pattern_name=pattern,
                            tpms_type=tpms_type,
                            cell_count=cell_count,
                            level_set=level_set,
                            thickness=thickness,
                            invert=invert
                        )
                    except Exception as e:
                        print("TPMS application error:", e, file=sys.stderr)

                full_mask = np.zeros_like(self.volume_data, dtype=np.uint8)
                full_mask[lower:upper+1] = mask_crop
                self.segmented_data = full_mask

                center_slice = min(self.volume_data.shape[0]//2, upper)
                self.root.after(0, lambda: self.display_slice(center_slice))
                messagebox.showinfo("Segmentation", f"Segmentation completed (largest component preserved, crop applied). Pattern: {pattern}{' (inverted)' if invert else ''}")
                self.root.after(0, lambda: self.show_3d_preview(self.segmented_data))
            except Exception as e:
                messagebox.showerror("Segmentation Error", str(e))
            finally:
                self.global_progress.stop()

        threading.Thread(target=_segment, daemon=True).start()

    def preview_segmentation(self):
        if self.volume_data is None:
            messagebox.showwarning("No volume", "Load a volume first")
            return

        try:
            self.hu_min = int(self.hu_min_var.get())
            self.hu_max = int(self.hu_max_var.get())
        except Exception:
            pass

        Z = self.volume_data.shape[0]
        lower = int(self.crop_lower)
        upper = int(self.crop_upper)
        if lower < 0: lower = 0
        if upper >= Z: upper = Z - 1
        if lower >= upper:
            messagebox.showwarning("Invalid crop", "Crop lower must be less than crop upper.")
            return

        raw_mask = (self.volume_data >= self.hu_min) & (self.volume_data <= self.hu_max)
        full_mask = np.zeros_like(self.volume_data, dtype=np.uint8)
        full_mask[lower:upper+1] = raw_mask[lower:upper+1].astype(np.uint8)
        self.show_3d_preview(full_mask)

    def _prepare_mesh_from_mask(self, mask):
        verts, faces, _, _ = measure.marching_cubes(mask.astype(float), level=0.5, spacing=self.image_spacing)
        faces_with_counts = np.hstack([np.full((faces.shape[0],1), 3, dtype=np.int64), faces.astype(np.int64)])
        faces_flat = faces_with_counts.ravel()
        return verts.astype(np.float64), faces_flat.astype(np.int64)

    def _close_preview_if_open(self):
        if getattr(self, "preview_process", None) is not None:
            try:
                if self.preview_process.is_alive():
                    self.preview_process.terminate()
                    self.preview_process.join(timeout=2.0)
            except Exception:
                pass
            finally:
                self.preview_process = None
                self.preview_verts = None
                self.preview_faces_flat = None

    def show_3d_preview(self, mask=None, auto_smooth=True):
        if self.volume_data is None:
            messagebox.showwarning("No volume", "Load a volume first")
            return

        if mask is None:
            mask = (self.volume_data >= self.hu_min) & (self.volume_data <= self.hu_max)

        mask = (mask > 0).astype(np.uint8)
        if not np.any(mask):
            messagebox.showwarning("3D Preview Error", "Cannot generate 3D preview: the resulting mask is empty.")
            return

        try:
            verts, faces_flat = self._prepare_mesh_from_mask(mask)
        except Exception as e:
            messagebox.showerror("3D Preview Error", f"marching_cubes failed: {e}")
            return

        pv_mesh = pv.PolyData(verts, faces_flat).clean()

        if auto_smooth:
            try:
                pv_mesh = pv_mesh.smooth(n_iter=15, relaxation_factor=0.05)
            except Exception:
                pass
            try:
                pv_mesh = pv_mesh.smooth_taubin(n_iter=30, pass_band=0.1)
            except Exception:
                pass

        try:
            pv_mesh = pv_mesh.fill_holes(1000.0)
        except Exception:
            pass

        pv_mesh = pv_mesh.clean()
        child_verts = pv_mesh.points.astype(np.float64)
        child_faces_flat = pv_mesh.faces.astype(np.int64)

        self.current_pv_mesh = pv_mesh
        try:
            self.current_tri = trimesh.Trimesh(vertices=pv_mesh.points, faces=pv_mesh.faces.reshape(-1,4)[:,1:], process=False)
        except Exception:
            self.current_tri = None

        self._close_preview_if_open()

        try:
            p = multiprocessing.Process(target=pv_viewer_process, args=(child_verts, child_faces_flat, self.preview_color))
            p.daemon = True
            p.start()
            self.preview_process = p
            self.preview_verts = child_verts
            self.preview_faces_flat = child_faces_flat
        except Exception as e:
            messagebox.showerror("Preview Spawn Error", str(e))

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
                try:
                    pv_mesh = pv_mesh.fill_holes(1000.0)
                except Exception:
                    pass
                pv_mesh = pv_mesh.clean()
                self.current_pv_mesh = pv_mesh

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

    def export_stl(self):
        if self.segmented_data is None and self.current_pv_mesh is None:
            messagebox.showwarning("No mesh", "No mesh available. Run segmentation or preview first.")
            return

        out_path = filedialog.asksaveasfilename(title="Save STL as", defaultextension=".stl", filetypes=[("STL", "*.stl")])
        if not out_path:
            return

        self._close_preview_if_open()
        progress = ProgressDialog(self.root, title="Exporting STL", maximum=100)

        def _export():
            try:
                def setp(pct, txt=None):
                    progress.update(value=pct, text=txt)

                setp(0, "Preparing mesh...")
                if self.current_pv_mesh is not None:
                    pv_mesh = self.current_pv_mesh.copy()
                else:
                    if self.segmented_data is None:
                        raise RuntimeError("No segmentation to export.")
                    if not np.any(self.segmented_data):
                        raise RuntimeError("Cannot export: the segmented mask is empty.")
                    verts, faces_flat = self._prepare_mesh_from_mask(self.segmented_data)
                    pv_mesh = pv.PolyData(verts, faces_flat)

                time.sleep(0.05)
                setp(10, "Surface extraction & cleaning...")
                try:
                    pv_mesh = pv_mesh.extract_surface().triangulate().clean(tolerance=1e-3)
                except Exception:
                    pv_mesh = pv_mesh.clean()

                time.sleep(0.05)
                setp(35, "Initial smoothing...")
                try:
                    pv_mesh = pv_mesh.smooth(n_iter=15, relaxation_factor=0.05)
                except Exception:
                    pass

                time.sleep(0.05)
                setp(55, "Taubin smoothing & hole filling...")
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

                try:
                    tri = trimesh.Trimesh(vertices=pv_mesh.points, faces=pv_mesh.faces.reshape(-1,4)[:,1:], process=False)
                except Exception as e:
                    faces = pv_mesh.faces.reshape(-1,4)[:,1:]
                    tri = trimesh.Trimesh(vertices=pv_mesh.points, faces=faces, process=False)

                try:
                    comps = tri.split(only_watertight=False)
                    if len(comps) > 1:
                        tri = max(comps, key=lambda c: len(c.faces))
                except Exception:
                    pass

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
                tri.export(out_path)

                setp(100, "Done")
                time.sleep(0.1)
                progress.close()
                messagebox.showinfo("Export Complete", f"Saved STL to: {out_path}")
            except Exception as e:
                try:
                    progress.close()
                except Exception:
                    pass
                messagebox.showerror("Export Error", str(e))

        threading.Thread(target=_export, daemon=True).start()

    def logout(self):
        confirm = messagebox.askyesno("Logout", "Do you want to logout?")
        if confirm:
            # close preview process if running
            self._close_preview_if_open()
            # restart the app by destroying and re-launching login
            self.root.destroy()
            # relaunch
            os.execv(sys.executable, [sys.executable] + sys.argv)
            
    def exit_app(self):
        # Close the preview process before shutting down the app
        self._close_preview_if_open()
        self.root.destroy()
        # Forcing a clean exit to ensure all threads/processes terminate
        sys.exit(0)


# ---------- Entry point ----------
def main():
    try:
        multiprocessing.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

    root = tk.Tk()
    
    # Set window as non-resizable
    root.resizable(False, False)

    # Apply ttk theme and colors
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except Exception:
        pass

    # Common color vars
    THEME_BG = '#087830'
    BTN_BG = '#0a5d2a'
    FG = '#ffffff'
    TEXTBOX_FIELD_BG = '#149347'
    SLIDER_TROUGH_COLOR = TEXTBOX_FIELD_BG
    SLIDER_THUMB_COLOR = FG

    # Style configurations (ttk has limited options for colors)
    style.configure('.', background=THEME_BG)
    style.configure('TLabel', background=THEME_BG, foreground=FG)
    style.configure('TFrame', background=THEME_BG)
    style.configure('TLabelframe', background=THEME_BG)
    style.configure('TLabelframe.Label', background=THEME_BG, foreground=FG)
    style.configure('TButton', background=BTN_BG, foreground=FG, borderwidth=1)
    style.map('TButton', background=[('active', '#0c9239')], foreground=[('active', FG)])
    style.configure('TCheckbutton', background=THEME_BG, foreground=FG)
    
    # Textbox/Combobox configuration
    style.configure('TEntry', fieldbackground=TEXTBOX_FIELD_BG, foreground=FG)
    style.configure('TCombobox', fieldbackground=TEXTBOX_FIELD_BG, background=THEME_BG, foreground=FG)
    
    # Slider Color Configuration (Horizontal.TScale is the default style name)
    style.configure('Horizontal.TScale', 
                    background=THEME_BG, 
                    troughcolor=SLIDER_TROUGH_COLOR, # Green-ish trough
                    sliderrelief='flat')
    style.map('Horizontal.TScale', 
              background=[('active', THEME_BG)], 
              slidercolor=[('active', SLIDER_THUMB_COLOR), ('!active', SLIDER_THUMB_COLOR)]) # White thumb

    # Progressbar
    style.configure('Horizontal.TProgressbar', background=TEXTBOX_FIELD_BG, troughcolor=TEXTBOX_FIELD_BG)
    
    # Configure the dropdown menu (needed for Combobox)
    root.option_add('*TCombobox*Listbox*Background', TEXTBOX_FIELD_BG)
    root.option_add('*TCombobox*Listbox*Foreground', FG)
    root.option_add('*TCombobox*Listbox*selectBackground', '#0c9239')
    root.option_add('*TCombobox*Listbox*selectForeground', FG)

    # Apply background to root window
    root.configure(bg=THEME_BG)

    # Show login dialog and verify credentials
    def show_login_and_start():
        login_win = tk.Toplevel(root)
        login_win.title('Login')
        login_win.geometry('360x180') # Adjusted height
        login_win.resizable(False, False)

        # center and set background
        login_win.transient(root)
        login_win.grab_set()
        login_win.configure(bg=THEME_BG)

        users = load_users()

        def attempt_login():
            user = username_var.get().strip()
            pwd = password_var.get()
            ok, msg = verify_user(user, pwd)
            if ok:
                login_win.destroy()
                # start app
                CTBoneSegmentation(root)
            else:
                status_label.config(text=msg)
                
        # --- Centered Inputs using a Frame and Grid ---
        input_frame = ttk.Frame(login_win)
        input_frame.pack(pady=(16, 4))
        input_frame.columnconfigure(1, weight=1)

        # Username
        ttk.Label(input_frame, text='Username:').grid(row=0, column=0, sticky='w', padx=(0, 5))
        username_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=username_var, width=20).grid(row=0, column=1, sticky='ew')

        # Password
        ttk.Label(input_frame, text='Password:').grid(row=1, column=0, sticky='w', pady=(8, 0), padx=(0, 5))
        password_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=password_var, show='*', width=20).grid(row=1, column=1, sticky='ew', pady=(8, 0))
        # --- End Centered Inputs ---

        status_label = ttk.Label(login_win, text='')
        status_label.pack(pady=(6,0))

        # Use tk.Button for more control over color in the Toplevel
        tk.Button(login_win, text='Login', command=attempt_login, 
                  bg=BTN_BG, fg=FG, activebackground='#0c9239', activeforeground=FG).pack(pady=12)

    show_login_and_start()

    root.mainloop()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    main()
