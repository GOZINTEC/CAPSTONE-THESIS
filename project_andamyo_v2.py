"""
mimics_tpms_qt.py
=================
CT Bone Segmentation & TPMS Scaffold Generator
PyQt5 Medical Workstation with Quad-Viewport MPR Layout

Rewritten from CAPSTONE2_tpms.py (Tkinter) into a professional PyQt5
application styled after Materialise Mimics / SurgiCase.

Features:
    - 2x2 Quad-Viewport: Axial, Coronal, Sagittal slices + embedded 3D view
    - Crosshair synchronization across all 2D views
    - Window/Level contrast control (right-click drag)
    - Semi-transparent mask overlay on CT slices
    - TPMS scaffold pattern generation (6 patterns, Skeletal/Shell)
    - Background threading for all heavy computation
    - Embedded PyVista 3D viewport via pyvistaqt
    - STL export with mesh repair pipeline
"""

import os
import sys
import time
import multiprocessing
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage import measure, morphology
import pyvista as pv
import trimesh

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QSplitter, QDockWidget, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QSlider, QCheckBox,
    QProgressBar, QFileDialog, QMessageBox, QGroupBox, QFrame,
    QScrollArea, QSizePolicy, QStatusBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect, QPoint
from PyQt5.QtGui import QImage, QPainter, QColor, QPen, QFont

from pyvistaqt import QtInteractor


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_WL = 400        # Window Level  (bone default)
DEFAULT_WW = 1800       # Window Width  (bone default)
BONE_COLOR = [0.95, 0.93, 0.90]  # RGB – bone-white / ivory for high contrast
SCAFFOLD_COLOR = [0.0, 0.82, 1.0]  # RGB – titanium / surgical implant cyan
OVERLAY_BONE_COLOR = [0.95, 0.93, 0.90]  # Host bone in overlay mode
DEFAULT_BONE_ALPHA = 0.35  # Default transparency for overlay bone

# HU preset tuples: (hu_min, hu_max, wl, ww)
HU_PRESETS = {
    "Bone":           (150,  3000, 400,  1800),
    "Cortical Bone":  (500,  3000, 1000, 2500),
    "Cancellous":     (150,  500,  325,  350),
    "Soft Tissue":    (-200, 400,  40,   400),
}

TPMS_PATTERNS = [
    "None", "Gyroid", "Primitive", "Diamond",
    "Lidinoid", "Split-P", "Schwarz-P",
]

# Viewport border colours matching Mimics convention
COLOR_CORONAL  = QColor(0, 200, 0)      # green
COLOR_AXIAL    = QColor(220, 40, 40)     # red
COLOR_SAGITTAL = QColor(60, 120, 255)    # blue
COLOR_3D       = QColor(0, 200, 200)     # cyan

# Mask overlay colour (semi-transparent orange)
MASK_RGBA = (255, 80, 0)   # RGB; alpha applied separately


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def apply_windowing(slice_data, wl, ww):
    """Apply Window/Level to a 2D HU array, returning uint8 grayscale."""
    vmin = wl - ww / 2.0
    vmax = wl + ww / 2.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    out = ((slice_data.astype(np.float32) - vmin) / (vmax - vmin) * 255.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def numpy_to_qimage_rgba(arr_rgba):
    """Convert an (H, W, 4) uint8 RGBA numpy array to a QImage."""
    h, w = arr_rgba.shape[:2]
    arr_rgba = np.ascontiguousarray(arr_rgba)
    return QImage(arr_rgba.data, w, h, w * 4,
                  QImage.Format_RGBA8888).copy()


def create_bone_mask_keep_largest(vol, hu_min, hu_max):
    """
    Create bone mask from HU volume, morphologically clean it,
    and keep only the largest connected component.
    """
    m = (vol >= hu_min) & (vol <= hu_max)
    m = morphology.remove_small_objects(m, min_size=1000)
    m = ndimage.binary_fill_holes(m)
    m = morphology.binary_closing(m, morphology.ball(2))
    labeled, ncomp = ndimage.label(m)
    if ncomp > 1:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        m = (labeled == np.argmax(counts))
    return m.astype(np.uint8)


def apply_tpms_to_crop(mask_crop, pattern_name, tpms_type,
                        cell_count, level_set, thickness, invert=False):
    """
    Apply a TPMS implicit-surface pattern to a cropped binary mask.

    Parameters
    ----------
    mask_crop    : 3-D uint8 array (Z, Y, X), 0/1 bone mask.
    pattern_name : One of TPMS_PATTERNS.
    tpms_type    : "Skeletal" or "Shell".
    cell_count   : int, number of unit-cell repeats across longest dim.
    level_set    : float, iso-value for Skeletal type (-2 … 2).
    thickness    : float, normalised wall thickness for Shell (0.01 … 0.5).
    invert       : bool, invert the solid/void phase.

    Returns
    -------
    Combined binary mask (uint8) of bone AND pattern.
    """
    if pattern_name in (None, "None"):
        return mask_crop

    mask_bool = mask_crop > 0
    if not np.any(mask_bool):
        return mask_crop

    zdim, ydim, xdim = mask_bool.shape
    cell_count = max(1, int(cell_count))

    # --- Normalised isotropic coordinate grid ---
    max_dim = max(zdim, ydim, xdim)
    zz = np.linspace(0.0, zdim / max_dim, zdim, endpoint=False)
    yy = np.linspace(0.0, ydim / max_dim, ydim, endpoint=False)
    xx = np.linspace(0.0, xdim / max_dim, xdim, endpoint=False)
    Z, Y, X = np.meshgrid(zz, yy, xx, indexing="ij")

    scale = 2.0 * np.pi * float(cell_count)
    Xs, Ys, Zs = X * scale, Y * scale, Z * scale

    # --- Evaluate implicit field ---
    if pattern_name == "Gyroid":
        field = (np.cos(Xs) * np.sin(Ys)
                 + np.cos(Ys) * np.sin(Zs)
                 + np.cos(Zs) * np.sin(Xs))

    elif pattern_name in ("Primitive", "Schwarz-P"):
        field = np.cos(Xs) + np.cos(Ys) + np.cos(Zs)

    elif pattern_name == "Diamond":
        field = (np.sin(Xs) * np.sin(Ys) * np.sin(Zs)
                 + np.sin(Xs) * np.cos(Ys) * np.cos(Zs)
                 + np.cos(Xs) * np.sin(Ys) * np.cos(Zs)
                 + np.cos(Xs) * np.cos(Ys) * np.sin(Zs))

    elif pattern_name == "Lidinoid":
        field = (np.sin(2*Xs)*np.cos(Ys)*np.sin(Zs)
                 + np.sin(Xs)*np.sin(2*Ys)*np.cos(Zs)
                 + np.cos(Xs)*np.sin(Ys)*np.sin(2*Zs)
                 - np.cos(2*Xs)*np.cos(2*Ys)
                 - np.cos(2*Ys)*np.cos(2*Zs)
                 - np.cos(2*Zs)*np.cos(2*Xs) + 0.3)

    elif pattern_name == "Split-P":
        field = (1.1 * (np.sin(2*Xs)*np.cos(Ys)*np.sin(Zs)
                        + np.sin(Xs)*np.sin(2*Ys)*np.cos(Zs)
                        + np.cos(Xs)*np.sin(Ys)*np.sin(2*Zs))
                 - 0.2 * (np.cos(2*Xs)*np.cos(2*Ys)
                          + np.cos(2*Ys)*np.cos(2*Zs)
                          + np.cos(2*Zs)*np.cos(2*Xs))
                 - 0.4 * (np.cos(2*Xs) + np.cos(2*Ys) + np.cos(2*Zs)))
    else:
        return mask_crop

    # --- Skeletal vs Shell ---
    if tpms_type == "Skeletal":
        pattern_mask = (field > level_set)

    elif tpms_type == "Shell":
        # Normalise field within bone voxels to [-0.5, 0.5]
        field_in = field[mask_bool]
        if field_in.size == 0:
            return np.zeros_like(mask_crop, dtype=np.uint8)
        f_min, f_max = field_in.min(), field_in.max()
        if (f_max - f_min) > 1e-5:
            field = (field - f_min) / (f_max - f_min) - 0.5
        else:
            field.fill(0.0)
        pattern_mask = (np.abs(field) < thickness)

    else:
        pattern_mask = (field > 0)

    if invert:
        pattern_mask = ~pattern_mask

    return np.logical_and(mask_bool, pattern_mask).astype(np.uint8)


def prepare_mesh_from_mask(mask, spacing):
    """
    Run marching cubes on a binary mask and return (verts, faces_flat)
    in the format required by pv.PolyData.
    """
    verts, faces, _, _ = measure.marching_cubes(
        mask.astype(float), level=0.5, spacing=spacing
    )
    faces_with_n = np.hstack([
        np.full((faces.shape[0], 1), 3, dtype=np.int64),
        faces.astype(np.int64),
    ])
    return verts.astype(np.float64), faces_with_n.ravel().astype(np.int64)


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER THREADS  (keep GUI responsive during heavy computation)
# ══════════════════════════════════════════════════════════════════════════════

class LoadVolumeWorker(QThread):
    """Background thread for loading DICOM series or NRRD volumes."""
    finished = pyqtSignal(object, tuple)   # (np.ndarray, spacing_zyx)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int, str)        # (percent, message)

    def __init__(self, dicom_folder=None, nrrd_path=None, parent=None):
        super().__init__(parent)
        self.dicom_folder = dicom_folder
        self.nrrd_path = nrrd_path

    def run(self):
        try:
            self.progress.emit(10, "Reading image data…")
            if self.nrrd_path:
                img = sitk.ReadImage(self.nrrd_path)
            elif self.dicom_folder:
                reader = sitk.ImageSeriesReader()
                paths = reader.GetGDCMSeriesFileNames(self.dicom_folder)
                if not paths:
                    raise RuntimeError("No DICOM series found in folder")
                reader.SetFileNames(paths)
                img = reader.Execute()
            else:
                raise RuntimeError("No input file specified")

            self.progress.emit(60, "Converting to NumPy array…")
            arr = sitk.GetArrayFromImage(img).astype(np.int16)  # (Z, Y, X)
            sp = img.GetSpacing()                                # (X, Y, Z)
            spacing_zyx = (sp[2], sp[1], sp[0])

            self.progress.emit(100, "Volume loaded")
            self.finished.emit(arr, spacing_zyx)

        except Exception as e:
            self.error.emit(str(e))


class SegmentWorker(QThread):
    """Background thread for bone segmentation + optional TPMS pattern.

    Produces BOTH the scaffold mesh (within the crop range) AND a host-bone
    mesh (the anatomical bone OUTSIDE the crop range) for overlay mode.
    """
    # (full_scaffold_mask, scaffold_pv_mesh, host_bone_pv_mesh_or_None)
    finished = pyqtSignal(object, object, object)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, volume, spacing, hu_min, hu_max,
                 crop_lo, crop_hi,
                 pattern, tpms_type, cell_count,
                 level_set, thickness, invert,
                 parent=None):
        super().__init__(parent)
        self.volume    = volume
        self.spacing   = spacing
        self.hu_min    = hu_min
        self.hu_max    = hu_max
        self.crop_lo   = crop_lo
        self.crop_hi   = crop_hi
        self.pattern   = pattern
        self.tpms_type = tpms_type
        self.cell_count = cell_count
        self.level_set = level_set
        self.thickness = thickness
        self.invert    = invert

    def run(self):
        try:
            Z = self.volume.shape[0]
            lo = max(0, self.crop_lo)
            hi = min(Z - 1, self.crop_hi)
            if lo >= hi:
                raise RuntimeError("Invalid crop range: lower must be < upper")

            self.progress.emit(10, "Thresholding & morphological filtering.")
            vol_crop = self.volume[lo:hi + 1]
            mask_crop = create_bone_mask_keep_largest(
                vol_crop, self.hu_min, self.hu_max
            )

            # Apply TPMS if requested
            scaffold_crop = mask_crop.copy()
            if self.pattern and self.pattern != "None":
                self.progress.emit(40, f"Applying TPMS pattern ({self.pattern})…")
                try:
                    scaffold_crop = apply_tpms_to_crop(
                        mask_crop, self.pattern, self.tpms_type,
                        self.cell_count, self.level_set,
                        self.thickness, self.invert,
                    )
                except Exception as e:
                    print(f"TPMS error (continuing with plain mask): {e}",
                          file=sys.stderr)
                    scaffold_crop = mask_crop.copy()

            # --- Build full-volume scaffold mask ---
            full_scaffold_mask = np.zeros_like(self.volume, dtype=np.uint8)
            full_scaffold_mask[lo:hi + 1] = scaffold_crop
            if not np.any(full_scaffold_mask):
                raise RuntimeError("Resulting mask is empty — check HU / crop")

            # --- Build scaffold mesh ---
            self.progress.emit(55, "Running marching cubes on scaffold…")
            verts, faces_flat = prepare_mesh_from_mask(
                full_scaffold_mask, self.spacing
            )
            scaffold_pv_mesh = pv.PolyData(verts, faces_flat).clean()

            self.progress.emit(65, "Smoothing scaffold mesh…")
            try:
                scaffold_pv_mesh = scaffold_pv_mesh.smooth(
                    n_iter=15, relaxation_factor=0.05
                )
            except Exception:
                pass
            try:
                scaffold_pv_mesh = scaffold_pv_mesh.smooth_taubin(
                    n_iter=30, pass_band=0.1
                )
            except Exception:
                pass
            try:
                scaffold_pv_mesh = scaffold_pv_mesh.fill_holes(1000.0)
            except Exception:
                pass
            scaffold_pv_mesh = scaffold_pv_mesh.clean()

            # --- Build host bone mesh for overlay ---
            # The host bone is the anatomical bone OUTSIDE the crop range,
            # i.e. the bone that remains after the osteotomy resection.
            # Slices [0:lo] and [hi+1:Z] contain the host bone.
            bone_pv_mesh = None
            self.progress.emit(75, "Building host bone mesh for overlay…")
            try:
                host_bone_mask = np.zeros_like(self.volume, dtype=np.uint8)

                # Process each side of the crop independently so that
                # both proximal and distal bone fragments are preserved
                # (a single keep_largest call would discard one side).

                # Lower side: slices 0 .. lo-1
                if lo > 0:
                    vol_below = self.volume[:lo]
                    mask_below = (
                        (vol_below >= self.hu_min) & (vol_below <= self.hu_max)
                    ).astype(np.uint8)
                    mask_below = morphology.remove_small_objects(
                        mask_below > 0, min_size=500
                    ).astype(np.uint8)
                    mask_below = ndimage.binary_fill_holes(mask_below).astype(
                        np.uint8
                    )
                    host_bone_mask[:lo] = mask_below

                # Upper side: slices hi+1 .. Z-1
                if hi < Z - 1:
                    vol_above = self.volume[hi + 1:]
                    mask_above = (
                        (vol_above >= self.hu_min) & (vol_above <= self.hu_max)
                    ).astype(np.uint8)
                    mask_above = morphology.remove_small_objects(
                        mask_above > 0, min_size=500
                    ).astype(np.uint8)
                    mask_above = ndimage.binary_fill_holes(mask_above).astype(
                        np.uint8
                    )
                    host_bone_mask[hi + 1:] = mask_above

                if np.any(host_bone_mask):
                    self.progress.emit(85, "Marching cubes on host bone…")
                    bv, bf = prepare_mesh_from_mask(
                        host_bone_mask, self.spacing
                    )
                    bone_pv_mesh = pv.PolyData(bv, bf).clean()
                    # Light smooth for visual quality
                    try:
                        bone_pv_mesh = bone_pv_mesh.smooth(
                            n_iter=10, relaxation_factor=0.04
                        )
                    except Exception:
                        pass
                    bone_pv_mesh = bone_pv_mesh.clean()
            except Exception as e:
                print(f"Host bone mesh error (continuing): {e}",
                      file=sys.stderr)
                bone_pv_mesh = None

            self.progress.emit(100, "Segmentation complete")
            self.finished.emit(full_scaffold_mask, scaffold_pv_mesh,
                               bone_pv_mesh)

        except Exception as e:
            self.error.emit(str(e))


class ExportSTLWorker(QThread):
    """Background thread for the full STL export pipeline."""
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, pv_mesh, segmented_data, spacing, out_path, parent=None):
        super().__init__(parent)
        self.pv_mesh = pv_mesh
        self.segmented_data = segmented_data
        self.spacing = spacing
        self.out_path = out_path

    def run(self):
        try:
            self.progress.emit(5, "Preparing mesh…")
            if self.pv_mesh is not None:
                mesh = self.pv_mesh.copy()
            else:
                if self.segmented_data is None or not np.any(self.segmented_data):
                    raise RuntimeError("No segmentation data to export")
                v, f = prepare_mesh_from_mask(self.segmented_data, self.spacing)
                mesh = pv.PolyData(v, f)

            self.progress.emit(15, "Surface extraction & cleaning…")
            try:
                mesh = mesh.extract_surface().triangulate().clean(tolerance=1e-3)
            except Exception:
                mesh = mesh.clean()

            self.progress.emit(35, "Smoothing…")
            try:
                mesh = mesh.smooth(n_iter=15, relaxation_factor=0.05)
            except Exception:
                pass

            self.progress.emit(50, "Taubin smoothing & hole filling…")
            try:
                mesh = mesh.smooth_taubin(n_iter=30, pass_band=0.1)
            except Exception:
                pass
            try:
                mesh = mesh.fill_holes(1000.0)
            except Exception:
                pass
            mesh = mesh.clean()

            self.progress.emit(70, "Converting to Trimesh & repairing…")
            tri = trimesh.Trimesh(
                vertices=mesh.points,
                faces=mesh.faces.reshape(-1, 4)[:, 1:],
                process=False,
            )
            # Keep largest component
            try:
                comps = tri.split(only_watertight=False)
                if len(comps) > 1:
                    tri = max(comps, key=lambda c: len(c.faces))
            except Exception:
                pass
            # Repair steps
            for fn in (trimesh.repair.fix_normals,
                       trimesh.repair.fill_holes,
                       trimesh.repair.fix_inversion,
                       trimesh.repair.broken_faces):
                try:
                    fn(tri)
                except Exception:
                    pass

            self.progress.emit(90, "Saving STL…")
            tri.export(self.out_path)

            self.progress.emit(100, "Export complete")
            self.finished.emit(self.out_path)

        except Exception as e:
            self.error.emit(str(e))


class SmoothMeshWorker(QThread):
    """Background thread for additional mesh smoothing iterations."""
    finished = pyqtSignal(object)   # pv.PolyData
    error    = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, pv_mesh, parent=None):
        super().__init__(parent)
        self.pv_mesh = pv_mesh

    def run(self):
        try:
            self.progress.emit(10, "Smoothing mesh…")
            mesh = self.pv_mesh.copy()
            try:
                mesh = mesh.smooth(n_iter=15, relaxation_factor=0.05)
            except Exception:
                pass
            self.progress.emit(50, "Taubin smoothing…")
            try:
                mesh = mesh.smooth_taubin(n_iter=30, pass_band=0.1)
            except Exception:
                pass
            try:
                mesh = mesh.fill_holes(1000.0)
            except Exception:
                pass
            mesh = mesh.clean()
            self.progress.emit(100, "Smoothing complete")
            self.finished.emit(mesh)
        except Exception as e:
            self.error.emit(str(e))



# ══════════════════════════════════════════════════════════════════════════════
#  SLICE VIEW WIDGET  (custom QWidget for 2D CT viewport)
# ══════════════════════════════════════════════════════════════════════════════

class SliceViewWidget(QWidget):
    """
    Renders a single orthogonal CT slice with:
      - Window/Level contrast (right-click drag)
      - Coloured crosshair lines synchronised with other views
      - Semi-transparent mask overlay
      - Anatomical orientation labels
      - Mouse-wheel slice scrolling
      - Click-to-navigate (left button)
    """
    # Signals
    slice_changed     = pyqtSignal(str, int)        # (plane, new_index)
    click_navigated   = pyqtSignal(str, int, int)   # (plane, h_coord, v_coord)
    wl_changed_sig    = pyqtSignal(int, int)         # (wl, ww) broadcast

    def __init__(self, plane, border_color, orientation_labels, parent=None):
        """
        Parameters
        ----------
        plane : str
            One of 'axial', 'coronal', 'sagittal'.
        border_color : QColor
            Colour of the viewport border frame.
        orientation_labels : dict
            Maps position → label, e.g. {'top':'S', 'bottom':'I', ...}
        """
        super().__init__(parent)
        self.plane = plane
        self.border_color = border_color
        self.ori_labels = orientation_labels

        # Volume references (set by MainWindow)
        self.volume_data = None   # int16  (Z, Y, X)
        self.mask_data   = None   # uint8  (Z, Y, X)

        # Slice state
        self.slice_index = 0
        self.max_index   = 0

        # Crosshair positions (image-space row, col of *this* view's 2D image)
        self.cross_h = 0   # horizontal line (row position)
        self.cross_v = 0   # vertical   line (col position)

        # Window / Level
        self.wl = DEFAULT_WL
        self.ww = DEFAULT_WW

        # Mask overlay
        self.show_mask    = True
        self.mask_opacity = 90     # 0–255

        # W/L right-click drag state
        self._wl_dragging   = False
        self._drag_origin   = QPoint()
        self._drag_wl_start = DEFAULT_WL
        self._drag_ww_start = DEFAULT_WW

        # Cached QImage for the current display
        self._qimage = None
        # Bounding rect where image is painted (for coordinate mapping)
        self._img_rect = (0, 0, 1, 1)   # (x, y, w, h)

        self.setMinimumSize(160, 160)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)

    # ── data setters ──────────────────────────────────────────────────────

    def set_volume(self, vol):
        self.volume_data = vol
        if vol is not None:
            self._recompute_max()
            self.slice_index = min(self.slice_index, self.max_index)
        self._rebuild_image()

    def set_mask(self, mask):
        self.mask_data = mask
        self._rebuild_image()

    def set_slice(self, idx):
        idx = max(0, min(idx, self.max_index))
        if idx != self.slice_index:
            self.slice_index = idx
            self._rebuild_image()

    def set_crosshairs(self, h, v):
        self.cross_h = h
        self.cross_v = v
        self.update()

    def set_wl_ww(self, wl, ww):
        self.wl = wl
        self.ww = max(1, ww)
        self._rebuild_image()

    # ── internal helpers ──────────────────────────────────────────────────

    def _recompute_max(self):
        if self.volume_data is None:
            self.max_index = 0
            return
        s = self.volume_data.shape
        if   self.plane == 'axial':    self.max_index = s[0] - 1
        elif self.plane == 'coronal':  self.max_index = s[1] - 1
        elif self.plane == 'sagittal': self.max_index = s[2] - 1

    def _extract_slice(self):
        """Return the 2-D HU slice for the current plane and index."""
        if self.volume_data is None:
            return None
        i = self.slice_index
        if   self.plane == 'axial':    return self.volume_data[i, :, :]
        elif self.plane == 'coronal':  return self.volume_data[:, i, :]
        elif self.plane == 'sagittal': return self.volume_data[:, :, i]

    def _extract_mask_slice(self):
        if self.mask_data is None:
            return None
        i = self.slice_index
        if   self.plane == 'axial':    return self.mask_data[i, :, :]
        elif self.plane == 'coronal':  return self.mask_data[:, i, :]
        elif self.plane == 'sagittal': return self.mask_data[:, :, i]

    def _rebuild_image(self):
        """Recompute the RGBA QImage from current slice, windowing, and mask."""
        raw = self._extract_slice()
        if raw is None:
            self._qimage = None
            self.update()
            return

        gray = apply_windowing(raw, self.wl, self.ww)
        h, w = gray.shape
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = gray
        rgba[:, :, 1] = gray
        rgba[:, :, 2] = gray
        rgba[:, :, 3] = 255

        # Mask overlay
        msl = self._extract_mask_slice()
        if msl is not None and self.show_mask:
            where = msl > 0
            if np.any(where):
                a = self.mask_opacity / 255.0
                ia = 1.0 - a
                rgba[where, 0] = np.clip(
                    rgba[where, 0].astype(np.float32) * ia + MASK_RGBA[0] * a,
                    0, 255).astype(np.uint8)
                rgba[where, 1] = np.clip(
                    rgba[where, 1].astype(np.float32) * ia + MASK_RGBA[1] * a,
                    0, 255).astype(np.uint8)
                rgba[where, 2] = np.clip(
                    rgba[where, 2].astype(np.float32) * ia + MASK_RGBA[2] * a,
                    0, 255).astype(np.uint8)

        self._qimage = numpy_to_qimage_rgba(rgba)
        self.update()

    def _image_dims(self):
        """Return (img_h, img_w) of the current slice image."""
        if self._qimage is None:
            return (1, 1)
        return (self._qimage.height(), self._qimage.width())

    def _compute_draw_rect(self):
        """Aspect-preserving bounding box for the image inside the widget."""
        if self._qimage is None:
            return (0, 0, self.width(), self.height())
        iw, ih = self._qimage.width(), self._qimage.height()
        ww, wh = self.width(), self.height()
        scale = min(ww / max(iw, 1), wh / max(ih, 1))
        dw = int(iw * scale)
        dh = int(ih * scale)
        dx = (ww - dw) // 2
        dy = (wh - dh) // 2
        return (dx, dy, dw, dh)

    def _widget_to_image(self, wx, wy):
        """Map a widget pixel coordinate to image (col, row)."""
        dx, dy, dw, dh = self._img_rect
        ih, iw = self._image_dims()
        col = (wx - dx) / max(dw, 1) * iw
        row = (wy - dy) / max(dh, 1) * ih
        return int(np.clip(col, 0, iw - 1)), int(np.clip(row, 0, ih - 1))

    def _image_to_widget(self, col, row):
        """Map image (col, row) to widget pixel coordinate."""
        dx, dy, dw, dh = self._img_rect
        ih, iw = self._image_dims()
        wx = dx + col / max(iw, 1) * dw
        wy = dy + row / max(ih, 1) * dh
        return int(wx), int(wy)

    # ── painting ──────────────────────────────────────────────────────────

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(0, 0, 0))

        rect = self._compute_draw_rect()
        self._img_rect = rect
        dx, dy, dw, dh = rect

        if self._qimage is not None:
            p.drawImage(QRect(dx, dy, dw, dh), self._qimage)

        # ── crosshairs ──
        cross_pen_h = QPen(QColor(0, 255, 0, 160), 1, Qt.DashLine)
        cross_pen_v = QPen(QColor(0, 200, 255, 160), 1, Qt.DashLine)

        # Horizontal crosshair (row → widget y)
        _, wy = self._image_to_widget(0, self.cross_h)
        p.setPen(cross_pen_h)
        p.drawLine(dx, wy, dx + dw, wy)

        # Vertical crosshair (col → widget x)
        wx, _ = self._image_to_widget(self.cross_v, 0)
        p.setPen(cross_pen_v)
        p.drawLine(wx, dy, wx, dy + dh)

        # ── orientation labels ──
        label_font = QFont("Arial", 11, QFont.Bold)
        p.setFont(label_font)
        p.setPen(QColor(255, 255, 255, 200))
        margin = 8
        if 'top' in self.ori_labels:
            p.drawText(dx + dw // 2 - 6, dy + margin + 14, self.ori_labels['top'])
        if 'bottom' in self.ori_labels:
            p.drawText(dx + dw // 2 - 6, dy + dh - margin, self.ori_labels['bottom'])
        if 'left' in self.ori_labels:
            p.drawText(dx + margin, dy + dh // 2 + 5, self.ori_labels['left'])
        if 'right' in self.ori_labels:
            p.drawText(dx + dw - margin - 12, dy + dh // 2 + 5, self.ori_labels['right'])

        # ── slice info text ──
        info_font = QFont("Consolas", 9)
        p.setFont(info_font)
        p.setPen(QColor(200, 200, 200))
        info = f"WL:{self.wl}  WW:{self.ww}"
        p.drawText(dx + 6, dy + dh - 6, info)

        # ── border ──
        p.setPen(QPen(self.border_color, 3))
        p.drawRect(self.rect().adjusted(1, 1, -2, -2))

        # ── slice number (top-left corner inside border) ──
        p.setPen(QColor(self.border_color))
        p.setFont(QFont("Consolas", 10, QFont.Bold))
        p.drawText(dx + 6, dy + 16,
                   f"{self.plane.capitalize()} #{self.slice_index}")
        p.end()

    # ── mouse / keyboard events ───────────────────────────────────────────

    def wheelEvent(self, ev):
        """Scroll slices with the mouse wheel."""
        delta = 1 if ev.angleDelta().y() < 0 else -1
        new_idx = max(0, min(self.slice_index + delta, self.max_index))
        if new_idx != self.slice_index:
            self.slice_index = new_idx
            self._rebuild_image()
            self.slice_changed.emit(self.plane, new_idx)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            col, row = self._widget_to_image(ev.x(), ev.y())
            self.click_navigated.emit(self.plane, row, col)
        elif ev.button() == Qt.RightButton:
            self._wl_dragging   = True
            self._drag_origin   = ev.pos()
            self._drag_wl_start = self.wl
            self._drag_ww_start = self.ww

    def mouseMoveEvent(self, ev):
        if self._wl_dragging:
            dx = ev.x() - self._drag_origin.x()
            dy = ev.y() - self._drag_origin.y()
            self.ww = max(1, self._drag_ww_start + dx * 4)
            self.wl = self._drag_wl_start - dy * 4
            self._rebuild_image()
            self.wl_changed_sig.emit(self.wl, self.ww)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.RightButton:
            self._wl_dragging = False


# ══════════════════════════════════════════════════════════════════════════════
#  VIEWPORT PANEL  (slice view + slider, or 3D interactor)
# ══════════════════════════════════════════════════════════════════════════════

class SliceViewportPanel(QFrame):
    """
    Composite widget that pairs a SliceViewWidget with a vertical QSlider
    along its right edge and a small slice-index label beneath.
    """
    def __init__(self, slice_view, parent=None):
        super().__init__(parent)
        self.slice_view = slice_view

        # Vertical slider (top = max index, bottom = 0 for natural scrolling)
        self.slider = QSlider(Qt.Vertical)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setInvertedAppearance(True)   # top = high index
        self.slider.setFixedWidth(18)
        self.slider.setStyleSheet("""
            QSlider::groove:vertical { background: #444; width: 6px; }
            QSlider::handle:vertical {
                background: #aaa; height: 14px; margin: 0 -4px;
                border-radius: 3px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.slice_view, 1)
        layout.addWidget(self.slider, 0)

        # Connect slider → view
        self.slider.valueChanged.connect(self._on_slider)
        # Connect view scroll → slider
        self.slice_view.slice_changed.connect(self._on_view_scroll)

    def _on_slider(self, val):
        self.slice_view.set_slice(val)
        self.slice_view.slice_changed.emit(self.slice_view.plane, val)

    def _on_view_scroll(self, _plane, idx):
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)

    def update_range(self, max_idx):
        self.slider.setMaximum(max_idx)


class ThreeDViewportPanel(QFrame):
    """Wraps a pyvistaqt.QtInteractor in a styled frame.

    Enhancements over original:
      - Studio 3-point light rig (key, fill, rim)
      - Eye-Dome Lighting (EDL) for depth-contour shading
      - Depth peeling for correct alpha-transparency compositing
      - Dual-actor management: scaffold (opaque) + overlay bone (translucent)
      - Real-time bone alpha control without re-meshing
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("border: 3px solid " + COLOR_3D.name() + ";")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)

        # Create the embedded PyVista interactor
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        # Mesh / actor state
        self.scaffold_mesh = None      # pv.PolyData
        self.bone_mesh = None          # pv.PolyData (host bone outside crop)
        self.scaffold_actor = None     # VTK actor ref
        self.bone_actor = None         # VTK actor ref
        self.overlay_enabled = False
        self.bone_alpha = DEFAULT_BONE_ALPHA

        # Initialise rendering environment
        self._init_lighting_and_rendering()

    # ── rendering setup ───────────────────────────────────────────────────

    def _init_lighting_and_rendering(self):
        """Configure background, EDL, depth peeling, and 3-point light rig."""
        # Dark gradient background (charcoal → near-black)
        try:
            self.plotter.set_background(color="#1a1c22", top="#0a0c10")
        except Exception:
            self.plotter.set_background("#1a1a1a")

        # Eye-Dome Lighting for depth-contour edge enhancement
        try:
            self.plotter.enable_eye_dome_lighting()
        except Exception:
            pass  # older GPU / driver fallback

        # Depth peeling for correct transparent-over-opaque compositing
        try:
            self.plotter.enable_depth_peeling(
                number_of_peels=4, occlusion_ratio=0.0
            )
        except Exception:
            pass  # fallback to standard alpha blending

        # Clear default lights and set up studio 3-point rig
        self.plotter.remove_all_lights()

        # Key light – bright, top-front-right, neutral white
        key_light = pv.Light(
            position=(1.0, 1.0, 2.0),
            focal_point=(0.0, 0.0, 0.0),
            color="white",
            intensity=0.85,
        )
        key_light.positional = False  # directional light
        self.plotter.add_light(key_light)

        # Fill light – softer, bottom-left, cool slate-blue tint
        fill_light = pv.Light(
            position=(-2.0, -1.0, -0.5),
            focal_point=(0.0, 0.0, 0.0),
            color="#B0C4DE",  # light steel blue
            intensity=0.35,
        )
        fill_light.positional = False
        self.plotter.add_light(fill_light)

        # Rim / back light – behind and above, for edge contour
        rim_light = pv.Light(
            position=(0.0, 2.0, -2.0),
            focal_point=(0.0, 0.0, 0.0),
            color="white",
            intensity=0.50,
        )
        rim_light.positional = False
        self.plotter.add_light(rim_light)

        self.plotter.add_axes()

    # ── public API ────────────────────────────────────────────────────────

    def display_mesh(self, pv_mesh, color=None):
        """Backwards-compatible: clear all and display a single mesh.

        Used by threshold preview and smoothing.
        """
        if color is None:
            color = BONE_COLOR
        self.plotter.clear()
        self.scaffold_actor = None
        self.bone_actor = None
        self.scaffold_mesh = None
        self.bone_mesh = None
        if pv_mesh is not None and pv_mesh.n_points > 0:
            self.scaffold_actor = self.plotter.add_mesh(
                pv_mesh, color=color, smooth_shading=True,
                ambient=0.20, diffuse=0.80,
                specular=0.35, specular_power=25,
                name="scaffold",
            )
        self.plotter.add_axes()
        self.plotter.reset_camera()

    def set_meshes(self, scaffold_mesh, bone_mesh=None):
        """Set both scaffold and overlay bone meshes and refresh the scene.

        This is the main entry point after segmentation completes.
        """
        self.scaffold_mesh = scaffold_mesh
        self.bone_mesh = bone_mesh

        self.plotter.clear()
        self.scaffold_actor = None
        self.bone_actor = None

        # Always show scaffold (opaque, implant cyan)
        if scaffold_mesh is not None and scaffold_mesh.n_points > 0:
            self.scaffold_actor = self.plotter.add_mesh(
                scaffold_mesh, color=SCAFFOLD_COLOR,
                smooth_shading=True,
                ambient=0.20, diffuse=0.80,
                specular=0.35, specular_power=25,
                name="scaffold",
            )

        # Add bone mesh (hidden unless overlay is enabled)
        if bone_mesh is not None and bone_mesh.n_points > 0:
            self.bone_actor = self.plotter.add_mesh(
                bone_mesh, color=OVERLAY_BONE_COLOR,
                smooth_shading=True,
                ambient=0.15, diffuse=0.75,
                specular=0.20, specular_power=15,
                opacity=self.bone_alpha if self.overlay_enabled else 0.0,
                name="overlay_bone",
            )
            # Hide if overlay not active
            if not self.overlay_enabled and self.bone_actor is not None:
                try:
                    self.bone_actor.SetVisibility(False)
                except Exception:
                    pass

        self.plotter.add_axes()
        self.plotter.reset_camera()

    def set_overlay_enabled(self, enabled):
        """Toggle overlay bone visibility."""
        self.overlay_enabled = enabled
        if self.bone_actor is not None:
            try:
                self.bone_actor.SetVisibility(enabled)
                if enabled:
                    self.bone_actor.GetProperty().SetOpacity(self.bone_alpha)
                self.plotter.render()
            except Exception:
                pass

    def set_bone_alpha(self, alpha):
        """Update bone transparency in real-time (no re-meshing)."""
        self.bone_alpha = alpha
        if self.bone_actor is not None and self.overlay_enabled:
            try:
                self.bone_actor.GetProperty().SetOpacity(alpha)
                self.plotter.render()
            except Exception:
                pass

    def reset_camera(self):
        """Reset the 3D camera to fit all visible actors."""
        try:
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception:
            pass

    def close_plotter(self):
        try:
            self.plotter.close()
        except Exception:
            pass



# ══════════════════════════════════════════════════════════════════════════════
#  COLLAPSIBLE GROUP BOX
# ══════════════════════════════════════════════════════════════════════════════

class CollapsibleGroupBox(QGroupBox):
    """A QGroupBox that can be collapsed/expanded by clicking its title."""
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.toggled.connect(self._on_toggle)
        self._content_height = 0

    def _on_toggle(self, checked):
        for child in self.children():
            if isinstance(child, QWidget):
                child.setVisible(checked)


# ══════════════════════════════════════════════════════════════════════════════
#  CONTROL PANEL  (right sidebar dock content)
# ══════════════════════════════════════════════════════════════════════════════

class ControlPanel(QScrollArea):
    """
    Right-hand sidebar containing all tool panels:
      1. Data Ingestion
      2. Threshold & Crop
      3. TPMS Scaffold
      4. Mesh Operations & Export
      5. 3D View & Anatomical Overlay
    """
    # Signals forwarded to MainWindow
    sig_select_dicom   = pyqtSignal()
    sig_select_nrrd    = pyqtSignal()
    sig_load_volume    = pyqtSignal()
    sig_preview_thresh = pyqtSignal()
    sig_segment        = pyqtSignal()
    sig_smooth_mesh    = pyqtSignal()
    sig_export_stl     = pyqtSignal()
    sig_overlay_toggled = pyqtSignal(bool)
    sig_alpha_changed   = pyqtSignal(float)
    sig_reset_camera    = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(280)
        self.setMaximumWidth(380)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        main_lay = QVBoxLayout(container)
        main_lay.setSpacing(6)

        # ── 1. Data Ingestion ─────────────────────────────────────────────
        grp_data = CollapsibleGroupBox("Data Ingestion")
        lay_data = QVBoxLayout()

        btn_row = QHBoxLayout()
        self.btn_dicom = QPushButton("DICOM Folder")
        self.btn_nrrd  = QPushButton("NRRD File")
        self.btn_dicom.clicked.connect(self.sig_select_dicom)
        self.btn_nrrd.clicked.connect(self.sig_select_nrrd)
        btn_row.addWidget(self.btn_dicom)
        btn_row.addWidget(self.btn_nrrd)
        lay_data.addLayout(btn_row)

        self.lbl_file = QLabel("No file selected")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet("color: #aaa;")
        lay_data.addWidget(self.lbl_file)

        self.lbl_dims = QLabel("")
        self.lbl_dims.setStyleSheet("color: #888; font-size: 10px;")
        lay_data.addWidget(self.lbl_dims)

        self.btn_load = QPushButton("⬇  Load Volume")
        self.btn_load.setStyleSheet(
            "background-color: #2a7; color: white; font-weight: bold; "
            "padding: 6px;"
        )
        self.btn_load.clicked.connect(self.sig_load_volume)
        lay_data.addWidget(self.btn_load)

        grp_data.setLayout(lay_data)
        main_lay.addWidget(grp_data)

        # ── 2. Threshold & Crop ───────────────────────────────────────────
        grp_thresh = CollapsibleGroupBox("Threshold && Crop")
        lay_t = QVBoxLayout()

        # HU min/max spinboxes
        hu_row = QHBoxLayout()
        hu_row.addWidget(QLabel("Min HU:"))
        self.spin_hu_min = QSpinBox()
        self.spin_hu_min.setRange(-2048, 10000)
        self.spin_hu_min.setValue(150)
        hu_row.addWidget(self.spin_hu_min)
        hu_row.addWidget(QLabel("Max HU:"))
        self.spin_hu_max = QSpinBox()
        self.spin_hu_max.setRange(-2048, 10000)
        self.spin_hu_max.setValue(3000)
        hu_row.addWidget(self.spin_hu_max)
        lay_t.addLayout(hu_row)

        # Preset buttons
        preset_row = QHBoxLayout()
        for name, (lo, hi, wl, ww) in HU_PRESETS.items():
            btn = QPushButton(name)
            btn.setFixedHeight(24)
            btn.setStyleSheet("font-size: 10px; padding: 2px 4px;")
            btn.clicked.connect(
                lambda _, lo_=lo, hi_=hi, wl_=wl, ww_=ww:
                    self._apply_preset(lo_, hi_, wl_, ww_)
            )
            preset_row.addWidget(btn)
        lay_t.addLayout(preset_row)

        # Z Crop sliders
        lay_t.addWidget(QLabel("Z-Crop Range:"))
        crop_row = QHBoxLayout()
        self.slider_crop_lo = QSlider(Qt.Horizontal)
        self.slider_crop_lo.setMinimum(0)
        self.slider_crop_lo.setMaximum(0)
        self.slider_crop_hi = QSlider(Qt.Horizontal)
        self.slider_crop_hi.setMinimum(0)
        self.slider_crop_hi.setMaximum(0)
        crop_row.addWidget(QLabel("Lo:"))
        crop_row.addWidget(self.slider_crop_lo, 1)
        crop_row.addWidget(QLabel("Hi:"))
        crop_row.addWidget(self.slider_crop_hi, 1)
        lay_t.addLayout(crop_row)

        self.lbl_crop = QLabel("Lo: 0  Hi: 0")
        self.lbl_crop.setStyleSheet("color: #aaa; font-size: 10px;")
        lay_t.addWidget(self.lbl_crop)

        self.slider_crop_lo.valueChanged.connect(self._on_crop_lo)
        self.slider_crop_hi.valueChanged.connect(self._on_crop_hi)

        self.btn_preview = QPushButton("Preview Threshold (3D)")
        self.btn_preview.clicked.connect(self.sig_preview_thresh)
        lay_t.addWidget(self.btn_preview)

        grp_thresh.setLayout(lay_t)
        main_lay.addWidget(grp_thresh)

        # ── 3. TPMS Scaffold ─────────────────────────────────────────────
        grp_tpms = CollapsibleGroupBox("TPMS Scaffold Pattern")
        lay_p = QVBoxLayout()

        # Pattern
        row_pat = QHBoxLayout()
        row_pat.addWidget(QLabel("Pattern:"))
        self.combo_pattern = QComboBox()
        self.combo_pattern.addItems(TPMS_PATTERNS)
        self.combo_pattern.setCurrentText("Gyroid")
        row_pat.addWidget(self.combo_pattern, 1)
        lay_p.addLayout(row_pat)

        # Invert
        self.chk_invert = QCheckBox("Invert Pattern")
        lay_p.addWidget(self.chk_invert)

        # Cell count
        row_cell = QHBoxLayout()
        row_cell.addWidget(QLabel("Cell Count:"))
        self.spin_cells = QSpinBox()
        self.spin_cells.setRange(1, 100)
        self.spin_cells.setValue(5)
        row_cell.addWidget(self.spin_cells)
        self.slider_cells = QSlider(Qt.Horizontal)
        self.slider_cells.setRange(1, 100)
        self.slider_cells.setValue(5)
        row_cell.addWidget(self.slider_cells, 1)
        lay_p.addLayout(row_cell)
        # sync spinbox ↔ slider
        self.spin_cells.valueChanged.connect(self.slider_cells.setValue)
        self.slider_cells.valueChanged.connect(self.spin_cells.setValue)

        # Type (Skeletal / Shell)
        row_type = QHBoxLayout()
        row_type.addWidget(QLabel("Type:"))
        self.combo_type = QComboBox()
        self.combo_type.addItems(["Skeletal", "Shell"])
        row_type.addWidget(self.combo_type, 1)
        lay_p.addLayout(row_type)

        # Level-set slider (Skeletal)
        row_level = QHBoxLayout()
        self.lbl_level_title = QLabel("Level-Set:")
        row_level.addWidget(self.lbl_level_title)
        self.slider_level = QSlider(Qt.Horizontal)
        self.slider_level.setRange(-200, 200)   # mapped as value / 100
        self.slider_level.setValue(0)
        row_level.addWidget(self.slider_level, 1)
        self.lbl_level_val = QLabel("0.00")
        self.lbl_level_val.setFixedWidth(40)
        row_level.addWidget(self.lbl_level_val)
        lay_p.addLayout(row_level)
        self.slider_level.valueChanged.connect(
            lambda v: self.lbl_level_val.setText(f"{v/100:.2f}")
        )

        # Thickness slider (Shell)
        row_thick = QHBoxLayout()
        self.lbl_thick_title = QLabel("Thickness:")
        row_thick.addWidget(self.lbl_thick_title)
        self.slider_thick = QSlider(Qt.Horizontal)
        self.slider_thick.setRange(1, 50)   # mapped as value / 100
        self.slider_thick.setValue(10)
        row_thick.addWidget(self.slider_thick, 1)
        self.lbl_thick_val = QLabel("0.10")
        self.lbl_thick_val.setFixedWidth(40)
        row_thick.addWidget(self.lbl_thick_val)
        lay_p.addLayout(row_thick)
        self.slider_thick.valueChanged.connect(
            lambda v: self.lbl_thick_val.setText(f"{v/100:.2f}")
        )

        # Enable/disable level-set vs thickness based on type
        self.combo_type.currentTextChanged.connect(self._on_type_changed)
        self._on_type_changed(self.combo_type.currentText())

        # Segment button
        self.btn_segment = QPushButton("⚙  Segment Bones")
        self.btn_segment.setStyleSheet(
            "background-color: #c63; color: white; font-weight: bold; "
            "padding: 6px;"
        )
        self.btn_segment.clicked.connect(self.sig_segment)
        lay_p.addWidget(self.btn_segment)

        grp_tpms.setLayout(lay_p)
        main_lay.addWidget(grp_tpms)

        # ── 4. Mesh Operations & Export ───────────────────────────────────
        grp_mesh = CollapsibleGroupBox("Mesh Operations && Export")
        lay_m = QVBoxLayout()

        self.btn_smooth = QPushButton("Smooth 3D Mesh")
        self.btn_smooth.clicked.connect(self.sig_smooth_mesh)
        lay_m.addWidget(self.btn_smooth)

        self.lbl_mesh_stats = QLabel("Vertices: –\nFaces: –\nWatertight: –")
        self.lbl_mesh_stats.setStyleSheet(
            "color: #aaa; font-size: 10px; font-family: Consolas;"
        )
        lay_m.addWidget(self.lbl_mesh_stats)

        self.btn_export = QPushButton("💾  Export STL")
        self.btn_export.setStyleSheet(
            "background-color: #37a; color: white; font-weight: bold; "
            "padding: 6px;"
        )
        self.btn_export.clicked.connect(self.sig_export_stl)
        lay_m.addWidget(self.btn_export)

        grp_mesh.setLayout(lay_m)
        main_lay.addWidget(grp_mesh)

        # ── 5. 3D View & Anatomical Overlay ──────────────────────────────
        grp_overlay = CollapsibleGroupBox("3D View && Anatomical Overlay")
        lay_o = QVBoxLayout()

        # Overlay toggle checkbox
        self.chk_overlay = QCheckBox("Overlay Host Bone")
        self.chk_overlay.setChecked(False)
        self.chk_overlay.setEnabled(False)  # disabled until segmentation
        self.chk_overlay.toggled.connect(self._on_overlay_toggled)
        lay_o.addWidget(self.chk_overlay)

        # Alpha slider
        row_alpha = QHBoxLayout()
        self.lbl_alpha_title = QLabel("Bone Transparency:")
        self.lbl_alpha_title.setEnabled(False)
        row_alpha.addWidget(self.lbl_alpha_title)
        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(0, 100)
        self.slider_alpha.setValue(int(DEFAULT_BONE_ALPHA * 100))
        self.slider_alpha.setEnabled(False)
        self.slider_alpha.valueChanged.connect(self._on_alpha_slider)
        row_alpha.addWidget(self.slider_alpha, 1)
        self.lbl_alpha_val = QLabel(f"{int(DEFAULT_BONE_ALPHA * 100)}%")
        self.lbl_alpha_val.setFixedWidth(40)
        self.lbl_alpha_val.setEnabled(False)
        row_alpha.addWidget(self.lbl_alpha_val)
        lay_o.addLayout(row_alpha)

        # Reset camera button
        self.btn_reset_cam = QPushButton("Reset 3D Camera")
        self.btn_reset_cam.clicked.connect(self.sig_reset_camera)
        lay_o.addWidget(self.btn_reset_cam)

        grp_overlay.setLayout(lay_o)
        main_lay.addWidget(grp_overlay)

        # Spacer
        main_lay.addStretch(1)

        self.setWidget(container)

    # ── helpers ───────────────────────────────────────────────────────────

    def _apply_preset(self, hu_lo, hu_hi, wl, ww):
        self.spin_hu_min.setValue(hu_lo)
        self.spin_hu_max.setValue(hu_hi)
        # Notify MainWindow to update W/L across all views
        self.wl_preset = (wl, ww)
        # Emit a synthetic signal by calling parent method if available
        main_win = self.window()
        if hasattr(main_win, 'apply_wl_preset'):
            main_win.apply_wl_preset(wl, ww)

    def _on_type_changed(self, text):
        skeletal = (text == "Skeletal")
        self.slider_level.setEnabled(skeletal)
        self.lbl_level_title.setEnabled(skeletal)
        self.lbl_level_val.setEnabled(skeletal)
        self.slider_thick.setEnabled(not skeletal)
        self.lbl_thick_title.setEnabled(not skeletal)
        self.lbl_thick_val.setEnabled(not skeletal)

    def _on_crop_lo(self, val):
        hi = self.slider_crop_hi.value()
        if val >= hi:
            val = max(0, hi - 1)
            self.slider_crop_lo.blockSignals(True)
            self.slider_crop_lo.setValue(val)
            self.slider_crop_lo.blockSignals(False)
        self.lbl_crop.setText(f"Lo: {val}  Hi: {hi}")

    def _on_crop_hi(self, val):
        lo = self.slider_crop_lo.value()
        if val <= lo:
            val = lo + 1
            self.slider_crop_hi.blockSignals(True)
            self.slider_crop_hi.setValue(val)
            self.slider_crop_hi.blockSignals(False)
        self.lbl_crop.setText(f"Lo: {lo}  Hi: {val}")

    def update_crop_range(self, max_z):
        """Called after a volume is loaded to set crop slider ranges."""
        self.slider_crop_lo.setMaximum(max_z)
        self.slider_crop_hi.setMaximum(max_z)
        self.slider_crop_lo.setValue(0)
        self.slider_crop_hi.setValue(max_z)
        self.lbl_crop.setText(f"Lo: 0  Hi: {max_z}")

    def update_mesh_stats(self, n_verts, n_faces, watertight):
        wt = "Yes" if watertight else "No"
        self.lbl_mesh_stats.setText(
            f"Vertices: {n_verts:,}\nFaces: {n_faces:,}\nWatertight: {wt}"
        )

    # Accessor properties for reading current TPMS settings
    @property
    def hu_min(self):
        return self.spin_hu_min.value()

    @property
    def hu_max(self):
        return self.spin_hu_max.value()

    @property
    def crop_lo(self):
        return self.slider_crop_lo.value()

    @property
    def crop_hi(self):
        return self.slider_crop_hi.value()

    @property
    def tpms_pattern(self):
        return self.combo_pattern.currentText()

    @property
    def tpms_type(self):
        return self.combo_type.currentText()

    @property
    def tpms_cells(self):
        return self.spin_cells.value()

    @property
    def tpms_level(self):
        return self.slider_level.value() / 100.0

    @property
    def tpms_thickness(self):
        return self.slider_thick.value() / 100.0

    @property
    def tpms_invert(self):
        return self.chk_invert.isChecked()

    # ── overlay helpers ───────────────────────────────────────────────────

    def _on_overlay_toggled(self, checked):
        """Forward overlay toggle to MainWindow."""
        self.sig_overlay_toggled.emit(checked)

    def _on_alpha_slider(self, val):
        """Forward alpha slider change to MainWindow."""
        self.lbl_alpha_val.setText(f"{val}%")
        self.sig_alpha_changed.emit(val / 100.0)

    def enable_overlay_controls(self, enabled):
        """Enable or disable the overlay controls (after segmentation)."""
        self.chk_overlay.setEnabled(enabled)
        self.slider_alpha.setEnabled(enabled)
        self.lbl_alpha_title.setEnabled(enabled)
        self.lbl_alpha_val.setEnabled(enabled)

    @property
    def overlay_mode(self):
        return self.chk_overlay.isChecked()

    @property
    def overlay_alpha(self):
        return self.slider_alpha.value() / 100.0



# ══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """
    Top-level application window implementing the Mimics-style
    quad-viewport medical imaging workstation.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT Bone Segmentation & TPMS Scaffold Tool")
        self.resize(1400, 900)

        # ── application state ─────────────────────────────────────────────
        self.volume_data    = None   # int16 (Z, Y, X)
        self.segmented_data = None   # uint8 (Z, Y, X)
        self.image_spacing  = (1.0, 1.0, 1.0)  # (Z, Y, X)
        self.current_pv_mesh = None  # pv.PolyData (scaffold)
        self.current_scaffold_mesh = None  # pv.PolyData
        self.current_bone_mesh = None      # pv.PolyData (host bone)

        # Current MPR cursor position (volume indices)
        self.cur_z = 0
        self.cur_y = 0
        self.cur_x = 0

        # File selection state
        self.dicom_folder = None
        self.nrrd_path    = None

        # Worker thread references
        self._worker = None

        # ── build UI ──────────────────────────────────────────────────────
        self._build_viewports()
        self._build_control_dock()
        self._build_status_bar()
        self._apply_stylesheet()

    # ── UI construction ───────────────────────────────────────────────────

    def _build_viewports(self):
        """Create the 2x2 quad-viewport grid as the central widget."""
        # Slice views
        self.view_coronal = SliceViewWidget(
            'coronal', COLOR_CORONAL,
            {'top': 'T', 'bottom': 'B', 'left': 'R', 'right': 'L'},
        )
        self.view_axial = SliceViewWidget(
            'axial', COLOR_AXIAL,
            {'top': 'A', 'bottom': 'P', 'left': 'R', 'right': 'L'},
        )
        self.view_sagittal = SliceViewWidget(
            'sagittal', COLOR_SAGITTAL,
            {'top': 'T', 'bottom': 'B', 'left': 'A', 'right': 'P'},
        )

        # Viewport panels (view + slider)
        self.vp_coronal  = SliceViewportPanel(self.view_coronal)
        self.vp_axial    = SliceViewportPanel(self.view_axial)
        self.vp_sagittal = SliceViewportPanel(self.view_sagittal)

        # 3D viewport
        self.vp_3d = ThreeDViewportPanel()

        # Assemble 2x2 with QSplitters
        top_split = QSplitter(Qt.Horizontal)
        top_split.addWidget(self.vp_coronal)
        top_split.addWidget(self.vp_axial)
        top_split.setSizes([500, 500])

        bot_split = QSplitter(Qt.Horizontal)
        bot_split.addWidget(self.vp_sagittal)
        bot_split.addWidget(self.vp_3d)
        bot_split.setSizes([500, 500])

        main_split = QSplitter(Qt.Vertical)
        main_split.addWidget(top_split)
        main_split.addWidget(bot_split)
        main_split.setSizes([450, 450])

        self.setCentralWidget(main_split)

        # ── Connect slice signals ─────────────────────────────────────────
        for view in (self.view_coronal, self.view_axial, self.view_sagittal):
            view.slice_changed.connect(self._on_slice_changed)
            view.click_navigated.connect(self._on_click_navigated)
            view.wl_changed_sig.connect(self._on_wl_changed_from_view)

    def _build_control_dock(self):
        """Create the right-side dockable control panel."""
        self.ctrl = ControlPanel()

        dock = QDockWidget("Controls", self)
        dock.setWidget(self.ctrl)
        dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        # Wire control panel signals
        self.ctrl.sig_select_dicom.connect(self._select_dicom_folder)
        self.ctrl.sig_select_nrrd.connect(self._select_nrrd_file)
        self.ctrl.sig_load_volume.connect(self._load_volume)
        self.ctrl.sig_preview_thresh.connect(self._preview_threshold)
        self.ctrl.sig_segment.connect(self._segment_bones)
        self.ctrl.sig_smooth_mesh.connect(self._smooth_mesh)
        self.ctrl.sig_export_stl.connect(self._export_stl)
        self.ctrl.sig_overlay_toggled.connect(self._on_toggle_overlay)
        self.ctrl.sig_alpha_changed.connect(self._on_overlay_alpha_changed)
        self.ctrl.sig_reset_camera.connect(self._on_reset_camera)

    def _build_status_bar(self):
        """Bottom status bar with progress indicator."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(260)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.status_bar.addPermanentWidget(self.progress_bar)

        self.lbl_status = QLabel("Ready")
        self.status_bar.addWidget(self.lbl_status, 1)

    def _apply_stylesheet(self):
        """Apply a dark medical-software theme."""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #2b2b2b;
                color: #ddd;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #555;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 14px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #444;
                border: 1px solid #666;
                border-radius: 3px;
                padding: 4px 10px;
                color: #ddd;
            }
            QPushButton:hover { background-color: #555; }
            QPushButton:pressed { background-color: #333; }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #383838;
                border: 1px solid #555;
                border-radius: 2px;
                padding: 2px 4px;
                color: #ddd;
            }
            QSlider::groove:horizontal {
                background: #444; height: 6px; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #aaa; width: 14px; margin: -4px 0;
                border-radius: 3px;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                background: #333;
                color: #ddd;
            }
            QProgressBar::chunk {
                background-color: #2a7;
                border-radius: 2px;
            }
            QScrollArea { border: none; }
            QStatusBar { background: #222; }
            QDockWidget {
                titlebar-close-icon: none;
                font-weight: bold;
            }
            QDockWidget::title {
                background: #333;
                padding: 6px;
            }
            QCheckBox { spacing: 6px; }
            QLabel { color: #ccc; }
        """)

    # ══════════════════════════════════════════════════════════════════════
    #  MPR CROSSHAIR SYNCHRONISATION
    # ══════════════════════════════════════════════════════════════════════

    def _sync_all_views(self):
        """Push current (cur_z, cur_y, cur_x) to all slice views."""
        if self.volume_data is None:
            return
        z, y, x = self.cur_z, self.cur_y, self.cur_x

        # Axial view: shows volume[z, :, :].  Crosshairs at (row=y, col=x)
        self.view_axial.set_slice(z)
        self.view_axial.set_crosshairs(y, x)
        self.vp_axial.slider.blockSignals(True)
        self.vp_axial.slider.setValue(z)
        self.vp_axial.slider.blockSignals(False)

        # Coronal view: shows volume[:, y, :].  Crosshairs at (row=z, col=x)
        self.view_coronal.set_slice(y)
        self.view_coronal.set_crosshairs(z, x)
        self.vp_coronal.slider.blockSignals(True)
        self.vp_coronal.slider.setValue(y)
        self.vp_coronal.slider.blockSignals(False)

        # Sagittal view: shows volume[:, :, x].  Crosshairs at (row=z, col=y)
        self.view_sagittal.set_slice(x)
        self.view_sagittal.set_crosshairs(z, y)
        self.vp_sagittal.slider.blockSignals(True)
        self.vp_sagittal.slider.setValue(x)
        self.vp_sagittal.slider.blockSignals(False)

    def _on_slice_changed(self, plane, idx):
        """Handle slice scroll / slider change from any view."""
        if plane == 'axial':
            self.cur_z = idx
        elif plane == 'coronal':
            self.cur_y = idx
        elif plane == 'sagittal':
            self.cur_x = idx
        self._sync_all_views()

    def _on_click_navigated(self, plane, row, col):
        """
        Handle left-click navigation on a 2D view.
        (row, col) are in image coordinates of the clicked view.
        """
        if plane == 'axial':
            # Image is Y×X → row=Y, col=X
            self.cur_y = row
            self.cur_x = col
        elif plane == 'coronal':
            # Image is Z×X → row=Z, col=X
            self.cur_z = row
            self.cur_x = col
        elif plane == 'sagittal':
            # Image is Z×Y → row=Z, col=Y
            self.cur_z = row
            self.cur_y = col
        self._sync_all_views()

    def _on_wl_changed_from_view(self, wl, ww):
        """Propagate W/L change from one view to all views."""
        for v in (self.view_axial, self.view_coronal, self.view_sagittal):
            v.set_wl_ww(wl, ww)

    def apply_wl_preset(self, wl, ww):
        """Called from ControlPanel preset buttons."""
        self._on_wl_changed_from_view(wl, ww)

    # ══════════════════════════════════════════════════════════════════════
    #  FILE SELECTION
    # ══════════════════════════════════════════════════════════════════════

    def _select_dicom_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if folder:
            self.dicom_folder = folder
            self.nrrd_path = None
            self.ctrl.lbl_file.setText(os.path.basename(folder))

    def _select_nrrd_file(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "Select NRRD File", "",
            "NRRD files (*.nrrd);;All files (*.*)"
        )
        if fp:
            self.nrrd_path = fp
            self.dicom_folder = None
            self.ctrl.lbl_file.setText(os.path.basename(fp))

    # ══════════════════════════════════════════════════════════════════════
    #  VOLUME LOADING
    # ══════════════════════════════════════════════════════════════════════

    def _load_volume(self):
        if not self.dicom_folder and not self.nrrd_path:
            QMessageBox.warning(self, "No Input",
                                "Select a DICOM folder or NRRD file first.")
            return

        self._set_busy(True)
        worker = LoadVolumeWorker(
            dicom_folder=self.dicom_folder,
            nrrd_path=self.nrrd_path,
        )
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_volume_loaded)
        worker.error.connect(self._on_worker_error)
        worker.finished.connect(lambda *_: self._set_busy(False))
        worker.error.connect(lambda *_: self._set_busy(False))
        self._worker = worker
        worker.start()

    def _on_volume_loaded(self, vol, spacing):
        self.volume_data    = vol
        self.image_spacing  = spacing
        self.segmented_data = None
        self.current_pv_mesh = None
        self.current_scaffold_mesh = None
        self.current_bone_mesh = None

        shape = vol.shape   # (Z, Y, X)
        self.ctrl.lbl_dims.setText(
            f"Dim: {shape[2]}×{shape[1]}×{shape[0]}  "
            f"Spacing: {spacing[2]:.3f}×{spacing[1]:.3f}×{spacing[0]:.3f} mm"
        )

        # Initialise cursor to volume centre
        self.cur_z = shape[0] // 2
        self.cur_y = shape[1] // 2
        self.cur_x = shape[2] // 2

        # Push volume to views
        for v in (self.view_axial, self.view_coronal, self.view_sagittal):
            v.set_volume(vol)
            v.set_mask(None)

        # Configure slider ranges
        self.vp_axial.update_range(shape[0] - 1)
        self.vp_coronal.update_range(shape[1] - 1)
        self.vp_sagittal.update_range(shape[2] - 1)

        # Crop range
        self.ctrl.update_crop_range(shape[0] - 1)

        # Sync views to centre
        self._sync_all_views()

        # Clear 3D viewport and disable overlay controls
        self.vp_3d.display_mesh(None)
        self.ctrl.enable_overlay_controls(False)
        self.ctrl.chk_overlay.setChecked(False)

        self.lbl_status.setText("Volume loaded successfully")

    # ══════════════════════════════════════════════════════════════════════
    #  THRESHOLD PREVIEW
    # ══════════════════════════════════════════════════════════════════════

    def _preview_threshold(self):
        """Show the raw HU threshold as a mask overlay on 2D views + 3D."""
        if self.volume_data is None:
            QMessageBox.warning(self, "No Volume", "Load a volume first.")
            return

        hu_min = self.ctrl.hu_min
        hu_max = self.ctrl.hu_max
        lo = self.ctrl.crop_lo
        hi = self.ctrl.crop_hi

        # Build mask (cropped region only)
        full_mask = np.zeros_like(self.volume_data, dtype=np.uint8)
        raw = (self.volume_data[lo:hi+1] >= hu_min) & \
              (self.volume_data[lo:hi+1] <= hu_max)
        full_mask[lo:hi+1] = raw.astype(np.uint8)

        self.segmented_data = full_mask

        # Update 2D overlays
        for v in (self.view_axial, self.view_coronal, self.view_sagittal):
            v.set_mask(full_mask)

        # Build and show 3D preview (if mask is non-empty)
        if np.any(full_mask):
            self._set_busy(True)
            self.lbl_status.setText("Building 3D threshold preview…")
            try:
                verts, faces_flat = prepare_mesh_from_mask(
                    full_mask, self.image_spacing
                )
                pv_mesh = pv.PolyData(verts, faces_flat).clean()
                self.current_pv_mesh = pv_mesh
                self.vp_3d.display_mesh(pv_mesh)
                self._update_mesh_stats(pv_mesh)
            except Exception as e:
                QMessageBox.warning(self, "3D Preview Error", str(e))
            finally:
                self._set_busy(False)
                self.lbl_status.setText("Threshold preview ready")

    # ══════════════════════════════════════════════════════════════════════
    #  SEGMENTATION
    # ══════════════════════════════════════════════════════════════════════

    def _segment_bones(self):
        if self.volume_data is None:
            QMessageBox.warning(self, "No Volume", "Load a volume first.")
            return

        self._set_busy(True)
        worker = SegmentWorker(
            volume     = self.volume_data,
            spacing    = self.image_spacing,
            hu_min     = self.ctrl.hu_min,
            hu_max     = self.ctrl.hu_max,
            crop_lo    = self.ctrl.crop_lo,
            crop_hi    = self.ctrl.crop_hi,
            pattern    = self.ctrl.tpms_pattern,
            tpms_type  = self.ctrl.tpms_type,
            cell_count = self.ctrl.tpms_cells,
            level_set  = self.ctrl.tpms_level,
            thickness  = self.ctrl.tpms_thickness,
            invert     = self.ctrl.tpms_invert,
        )
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_segment_done)
        worker.error.connect(self._on_worker_error)
        worker.finished.connect(lambda *_: self._set_busy(False))
        worker.error.connect(lambda *_: self._set_busy(False))
        self._worker = worker
        worker.start()

    def _on_segment_done(self, full_mask, scaffold_mesh, bone_mesh):
        self.segmented_data  = full_mask
        self.current_pv_mesh = scaffold_mesh       # backwards compat
        self.current_scaffold_mesh = scaffold_mesh
        self.current_bone_mesh = bone_mesh

        # Update mask overlay on all 2D views
        for v in (self.view_axial, self.view_coronal, self.view_sagittal):
            v.set_mask(full_mask)

        # Display meshes in 3D viewport (scaffold + overlay bone)
        self.vp_3d.set_meshes(scaffold_mesh, bone_mesh)
        self._update_mesh_stats(scaffold_mesh)

        # Enable overlay controls now that we have both meshes
        has_bone = bone_mesh is not None and bone_mesh.n_points > 0
        self.ctrl.enable_overlay_controls(has_bone)

        self.lbl_status.setText(
            f"Segmentation complete  |  Pattern: {self.ctrl.tpms_pattern}"
        )

    # ══════════════════════════════════════════════════════════════════════
    #  MESH SMOOTHING
    # ══════════════════════════════════════════════════════════════════════

    def _smooth_mesh(self):
        if self.current_pv_mesh is None:
            QMessageBox.warning(self, "No Mesh",
                                "Generate a 3D preview or segmentation first.")
            return

        self._set_busy(True)
        worker = SmoothMeshWorker(self.current_pv_mesh)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_smooth_done)
        worker.error.connect(self._on_worker_error)
        worker.finished.connect(lambda *_: self._set_busy(False))
        worker.error.connect(lambda *_: self._set_busy(False))
        self._worker = worker
        worker.start()

    def _on_smooth_done(self, pv_mesh):
        self.current_pv_mesh = pv_mesh
        self.current_scaffold_mesh = pv_mesh
        # Re-display with bone overlay preserved
        self.vp_3d.set_meshes(pv_mesh, self.current_bone_mesh)
        self._update_mesh_stats(pv_mesh)
        self.lbl_status.setText("Mesh smoothing complete")

    # ══════════════════════════════════════════════════════════════════════
    #  3D OVERLAY CONTROLS
    # ══════════════════════════════════════════════════════════════════════

    def _on_toggle_overlay(self, checked):
        """Toggle overlay bone visibility in the 3D viewport."""
        self.vp_3d.set_overlay_enabled(checked)

    def _on_overlay_alpha_changed(self, alpha):
        """Update bone transparency in real-time."""
        self.vp_3d.set_bone_alpha(alpha)

    def _on_reset_camera(self):
        """Reset the 3D camera to fit all visible actors."""
        self.vp_3d.reset_camera()

    # ══════════════════════════════════════════════════════════════════════
    #  STL EXPORT
    # ══════════════════════════════════════════════════════════════════════

    def _export_stl(self):
        if self.segmented_data is None and self.current_pv_mesh is None:
            QMessageBox.warning(self, "No Data",
                                "No mesh available. Run segmentation first.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save STL As", "", "STL files (*.stl)"
        )
        if not out_path:
            return

        self._set_busy(True)
        worker = ExportSTLWorker(
            pv_mesh=self.current_pv_mesh,
            segmented_data=self.segmented_data,
            spacing=self.image_spacing,
            out_path=out_path,
        )
        worker.progress.connect(self._on_progress)
        worker.finished.connect(
            lambda path: (
                self.lbl_status.setText(f"Exported: {path}"),
                QMessageBox.information(self, "Export Complete",
                                        f"STL saved to:\n{path}"),
            )
        )
        worker.error.connect(self._on_worker_error)
        worker.finished.connect(lambda *_: self._set_busy(False))
        worker.error.connect(lambda *_: self._set_busy(False))
        self._worker = worker
        worker.start()

    # ══════════════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.lbl_status.setText(msg)

    def _on_worker_error(self, msg):
        self.lbl_status.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Error", msg)

    def _set_busy(self, busy):
        """Enable/disable controls during background operations."""
        self.ctrl.btn_load.setEnabled(not busy)
        self.ctrl.btn_segment.setEnabled(not busy)
        self.ctrl.btn_smooth.setEnabled(not busy)
        self.ctrl.btn_export.setEnabled(not busy)
        if not busy:
            self.progress_bar.setValue(0)

    def _update_mesh_stats(self, pv_mesh):
        if pv_mesh is None:
            self.ctrl.update_mesh_stats(0, 0, False)
            return
        try:
            tri = trimesh.Trimesh(
                vertices=pv_mesh.points,
                faces=pv_mesh.faces.reshape(-1, 4)[:, 1:],
                process=False,
            )
            wt = tri.is_watertight
        except Exception:
            wt = False
        self.ctrl.update_mesh_stats(pv_mesh.n_points, pv_mesh.n_cells, wt)

    def closeEvent(self, event):
        """Clean up the 3D plotter before closing."""
        self.vp_3d.close_plotter()
        event.accept()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Windows DPI awareness
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")   # consistent cross-platform look

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    main()
