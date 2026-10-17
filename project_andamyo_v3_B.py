"""
project_andamyo_v3_B.py
=======================
CT Bone Segmentation, TPMS & Strut-Lattice Scaffold Generator (Version B)
PyQt5 Medical Workstation with Quad-Viewport MPR Layout

Version B Enhancements (Includes all Version A modifications + Strut Lattices):
    - 2D Viewport Reorientation in -90° / +90° increments for Axial, Coronal,
      and Sagittal views (preserves 3D scan coordinates while dynamically
      rotating 2D slices, crosshairs, click-navigation, and anatomical labels)
    - Medically Viable Physical Pore Size Engine (50 µm to 1500 µm) replacing
      dimensionless cell count, calibrated using true physical CT voxel spacing
      (in mm) with Nyquist resolution checks
    - Strut-Based Lattice Architectures via Periodic Signed Distance Fields (SDF):
        * Cubic (Simple Cubic)
        * BCC   (Body-Centered Cubic)
        * BCCZ  (Body-Centered Cubic with Z-reinforcement struts)
        * FCC   (Face-Centered Cubic)
    - Dynamic UI parameter switching between TPMS (Skeletal/Shell, Level-Set,
      Wall Thickness) and Strut Lattices (Strut Diameter 50–800 µm)
    - Default scaffold pattern set to "None"
    - 2x2 Quad-Viewport: Axial, Coronal, Sagittal slices + embedded 3D view
    - Crosshair synchronization across all 2D views
    - Window/Level contrast control (right-click drag)
    - Semi-transparent mask overlay on CT slices
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

# Pore size configuration (micrometers)
MIN_PORE_UM = 50
MAX_PORE_UM = 1500
DEFAULT_PORE_UM = 500
STEP_PORE_UM = 25

# Strut diameter configuration (micrometers)
MIN_STRUT_UM = 50
MAX_STRUT_UM = 800
DEFAULT_STRUT_UM = 200
STEP_STRUT_UM = 25

# HU preset tuples: (hu_min, hu_max, wl, ww)
HU_PRESETS = {
    "Bone":           (150,  3000, 400,  1800),
    "Cortical Bone":  (500,  3000, 1000, 2500),
    "Cancellous":     (150,  500,  325,  350),
    "Soft Tissue":    (-200, 400,  40,   400),
}

TPMS_PATTERNS = [
    "Gyroid", "Primitive", "Diamond",
    "Lidinoid", "Split-P", "Schwarz-P",
]

STRUT_PATTERNS = [
    "Cubic", "BCC", "BCCZ", "FCC",
]

ALL_SCAFFOLD_PATTERNS = ["None"] + TPMS_PATTERNS + STRUT_PATTERNS

# Empirical geometric factors relating TPMS unit cell period L to hydraulic pore diameter D_p
TPMS_ALPHA_FACTORS = {
    "Gyroid":    1.35,
    "Primitive": 1.25,
    "Schwarz-P": 1.25,
    "Diamond":   1.45,
    "Lidinoid":  1.60,
    "Split-P":   1.50,
}

# Viewport border colours matching Mimics convention
COLOR_CORONAL  = QColor(0, 200, 0)      # green
COLOR_AXIAL    = QColor(220, 40, 40)     # red
COLOR_SAGITTAL = QColor(60, 120, 255)    # blue
COLOR_3D       = QColor(0, 200, 200)     # cyan

# Mask overlay colour (semi-transparent orange)
MASK_RGBA = (255, 80, 0)   # RGB; alpha applied separately


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY & LATTICE GENERATION FUNCTIONS
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


def is_strut_pattern(pattern_name):
    """Return True if pattern_name is one of the strut-based lattice patterns."""
    return pattern_name in STRUT_PATTERNS


def dist_segment(px, py, pz, ax, ay, az, bx, by, bz):
    """
    Analytical Euclidean distance from 3D point array (px, py, pz)
    to a finite 3D line segment with endpoints A=(ax, ay, az) and B=(bx, by, bz).
    """
    abx = bx - ax
    aby = by - ay
    abz = bz - az
    ab_len_sq = abx * abx + aby * aby + abz * abz
    if ab_len_sq < 1e-12:
        return np.sqrt((px - ax) ** 2 + (py - ay) ** 2 + (pz - az) ** 2)

    apx = px - ax
    apy = py - ay
    apz = pz - az
    t = (apx * abx + apy * aby + apz * abz) / ab_len_sq
    t = np.clip(t, 0.0, 1.0)

    cx = ax + t * abx
    cy = ay + t * aby
    cz = az + t * abz
    return np.sqrt((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2)


def compute_unit_cell_period_mm(pattern_name, tpms_type, pore_size_um,
                                level_set=0.0, thickness=0.10,
                                strut_diam_um=DEFAULT_STRUT_UM):
    """
    Compute the physical unit-cell period L (in mm) required to achieve
    the target pore size (in micrometers) for either TPMS or Strut lattices.
    """
    dp_mm = max(0.01, float(pore_size_um) / 1000.0)
    d_strut_mm = max(0.01, float(strut_diam_um) / 1000.0)

    if is_strut_pattern(pattern_name):
        # Strut lattice geometric relationship between clear pore opening D_p,
        # strut diameter d_strut, and unit cell period L:
        if pattern_name == "Cubic":
            # Clear distance between parallel edge struts is L - d_strut = D_p
            L_mm = dp_mm + d_strut_mm
        elif pattern_name in ("BCC", "BCCZ"):
            # Body diagonals intersect at cell center; effective pore opening
            L_mm = 1.25 * dp_mm + d_strut_mm * np.sqrt(2.0)
        elif pattern_name == "FCC":
            # Face diagonals cross on each face; clear pore opening scales by sqrt(2)
            L_mm = np.sqrt(2.0) * dp_mm + d_strut_mm
        else:
            L_mm = dp_mm + d_strut_mm
        return max(0.05, float(L_mm))

    # TPMS pattern
    alpha = TPMS_ALPHA_FACTORS.get(pattern_name, 1.35)
    if tpms_type == "Shell":
        void_fraction = max(0.25, 1.0 - 1.5 * float(thickness))
        L_mm = (alpha * dp_mm) / void_fraction
    else:
        void_mod = np.clip(1.0 - 0.15 * float(level_set), 0.55, 1.65)
        L_mm = alpha * dp_mm * void_mod

    return max(0.05, float(L_mm))


def evaluate_strut_lattice_sdf(X_mm, Y_mm, Z_mm, pattern_name, L_mm):
    """
    Evaluate the Periodic Signed Distance Field (distance to nearest strut axis)
    for Cubic, BCC, BCCZ, or FCC unit cells of period L_mm.

    Exploits octahedral reflection symmetry across the unit cell center:
    Mapping periodic coordinates into the positive octant [0, L/2]^3
    allows exact analytical segment distance evaluation without redundant
    loops over all 8 octants.
    """
    h = 0.5 * L_mm
    # Map physical coordinates into centered periodic cell [-L/2, L/2]^3
    # and fold into the symmetric positive octant [0, h]^3
    u = np.abs(np.mod(X_mm, L_mm) - h)
    v = np.abs(np.mod(Y_mm, L_mm) - h)
    w = np.abs(np.mod(Z_mm, L_mm) - h)

    if pattern_name == "Cubic":
        # 12 edges of the unit cube [-h, h]^3 + 3 central orthogonal axes
        # In the folded octant [0, h]^3, the cube frame struts lie along:
        #   1) X-parallel edge (v = h, w = h) and central X-axis (v = 0, w = 0)
        #   2) Y-parallel edge (u = h, w = h) and central Y-axis (u = 0, w = 0)
        #   3) Z-parallel edge (u = h, v = h) and central Z-axis (u = 0, v = 0)
        d_edge_x = dist_segment(u, v, w, 0.0, h, h, h, h, h)
        d_edge_y = dist_segment(u, v, w, h, 0.0, h, h, h, h)
        d_edge_z = dist_segment(u, v, w, h, h, 0.0, h, h, h)
        d_axis_x = dist_segment(u, v, w, 0.0, 0.0, 0.0, h, 0.0, 0.0)
        d_axis_y = dist_segment(u, v, w, 0.0, 0.0, 0.0, 0.0, h, 0.0)
        d_axis_z = dist_segment(u, v, w, 0.0, 0.0, 0.0, 0.0, 0.0, h)
        return np.minimum.reduce([
            d_edge_x, d_edge_y, d_edge_z,
            d_axis_x, d_axis_y, d_axis_z
        ])

    elif pattern_name == "BCC":
        # 4 body diagonals connecting (-h,-h,-h)..(h,h,h) through origin (0,0,0)
        # In the positive octant [0, h]^3, the closest body diagonal is (0,0,0) -> (h,h,h)
        return dist_segment(u, v, w, 0.0, 0.0, 0.0, h, h, h)

    elif pattern_name == "BCCZ":
        # BCC body diagonals + vertical Z-reinforcement struts (along cell Z-edges and Z-center)
        d_bcc = dist_segment(u, v, w, 0.0, 0.0, 0.0, h, h, h)
        d_z_edge = dist_segment(u, v, w, h, h, 0.0, h, h, h)
        d_z_center = dist_segment(u, v, w, 0.0, 0.0, 0.0, 0.0, 0.0, h)
        return np.minimum(d_bcc, np.minimum(d_z_edge, d_z_center))

    elif pattern_name == "FCC":
        # Face-Centered Cubic: struts along face diagonals on the cube faces
        # plus interior face-center-to-vertex diagonals
        # Face x = h: diagonal (h, 0, 0) <-> (h, h, h) and (h, h, 0) <-> (h, 0, h)
        d_fx1 = dist_segment(u, v, w, h, 0.0, 0.0, h, h, h)
        d_fx2 = dist_segment(u, v, w, h, h, 0.0, h, 0.0, h)
        # Face y = h: diagonal (0, h, 0) <-> (h, h, h) and (h, h, 0) <-> (0, h, h)
        d_fy1 = dist_segment(u, v, w, 0.0, h, 0.0, h, h, h)
        d_fy2 = dist_segment(u, v, w, h, h, 0.0, 0.0, h, h)
        # Face z = h: diagonal (0, 0, h) <-> (h, h, h) and (h, 0, h) <-> (0, h, h)
        d_fz1 = dist_segment(u, v, w, 0.0, 0.0, h, h, h, h)
        d_fz2 = dist_segment(u, v, w, h, 0.0, h, 0.0, h, h)
        return np.minimum.reduce([d_fx1, d_fx2, d_fy1, d_fy2, d_fz1, d_fz2])

    return np.zeros_like(u)


def apply_scaffold_to_crop(mask_crop, pattern_name, tpms_type,
                           pore_size_um, level_set, thickness,
                           strut_diam_um=DEFAULT_STRUT_UM,
                           spacing=(1.0, 1.0, 1.0), invert=False):
    """
    Apply either a TPMS implicit-surface pattern or a Strut-Based Lattice
    pattern (Cubic, BCC, BCCZ, FCC) to a cropped binary bone mask using
    true physical coordinates (mm).

    Parameters
    ----------
    mask_crop     : 3-D uint8 array (Z, Y, X), 0/1 bone mask.
    pattern_name  : One of ALL_SCAFFOLD_PATTERNS.
    tpms_type     : "Skeletal" or "Shell" (for TPMS patterns).
    pore_size_um  : float, target pore size in micrometers (50 … 1500 µm).
    level_set     : float, iso-value for Skeletal TPMS (-2 … 2).
    thickness     : float, normalised wall thickness for Shell TPMS (0.01 … 0.5).
    strut_diam_um : float, strut diameter in micrometers for Strut patterns (50 … 800 µm).
    spacing       : tuple of (sz, sy, sx) voxel spacing in millimeters.
    invert        : bool, invert the solid/void phase.

    Returns
    -------
    Combined binary mask (uint8) of bone AND scaffold pattern.
    """
    if pattern_name in (None, "None"):
        return mask_crop

    mask_bool = mask_crop > 0
    if not np.any(mask_bool):
        return mask_crop

    zdim, ydim, xdim = mask_bool.shape
    sz, sy, sx = (float(spacing[0]), float(spacing[1]), float(spacing[2]))

    # Physical unit cell period L (mm)
    L_mm = compute_unit_cell_period_mm(
        pattern_name, tpms_type, pore_size_um,
        level_set=level_set, thickness=thickness,
        strut_diam_um=strut_diam_um,
    )

    # Physical coordinate axes in millimeters
    zz_mm = np.arange(zdim, dtype=np.float32) * sz
    yy_mm = np.arange(ydim, dtype=np.float32) * sy
    xx_mm = np.arange(xdim, dtype=np.float32) * sx

    # ── Branch 1: Strut-Based Lattice Architecture (Periodic SDF) ────────
    if is_strut_pattern(pattern_name):
        r_strut_mm = max(0.025, float(strut_diam_um) / 2000.0)
        pattern_mask = np.zeros(mask_bool.shape, dtype=bool)

        # Evaluate in Z-chunks to keep peak RAM usage low on high-res CT volumes
        chunk_z = max(1, min(32, zdim))
        for z0 in range(0, zdim, chunk_z):
            z1 = min(zdim, z0 + chunk_z)
            if not np.any(mask_bool[z0:z1]):
                continue
            Z_c, Y_c, X_c = np.meshgrid(
                zz_mm[z0:z1], yy_mm, xx_mm, indexing="ij"
            )
            d_min = evaluate_strut_lattice_sdf(
                X_c, Y_c, Z_c, pattern_name, L_mm
            )
            pattern_mask[z0:z1] = (d_min <= r_strut_mm)

        if invert:
            pattern_mask = ~pattern_mask

        return np.logical_and(mask_bool, pattern_mask).astype(np.uint8)

    # ── Branch 2: TPMS Implicit Surface Architecture ─────────────────────
    omega = (2.0 * np.pi) / L_mm
    Zs_1d = zz_mm * omega
    Ys_1d = yy_mm * omega
    Xs_1d = xx_mm * omega

    Zs, Ys, Xs = np.meshgrid(Zs_1d, Ys_1d, Xs_1d, indexing="ij")

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

    if tpms_type == "Skeletal":
        pattern_mask = (field > level_set)

    elif tpms_type == "Shell":
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
    """Background thread for bone segmentation + TPMS or Strut-Lattice pattern.

    Produces BOTH the scaffold mesh (within the crop range) AND a host-bone
    mesh (the anatomical bone OUTSIDE the crop range) for overlay mode.
    """
    # (full_scaffold_mask, scaffold_pv_mesh, host_bone_pv_mesh_or_None)
    finished = pyqtSignal(object, object, object)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, volume, spacing, hu_min, hu_max,
                 crop_lo, crop_hi,
                 pattern, tpms_type, pore_size_um,
                 level_set, thickness, strut_diam_um, invert,
                 parent=None):
        super().__init__(parent)
        self.volume        = volume
        self.spacing       = spacing
        self.hu_min        = hu_min
        self.hu_max        = hu_max
        self.crop_lo       = crop_lo
        self.crop_hi       = crop_hi
        self.pattern       = pattern
        self.tpms_type     = tpms_type
        self.pore_size_um  = pore_size_um
        self.level_set     = level_set
        self.thickness     = thickness
        self.strut_diam_um = strut_diam_um
        self.invert        = invert

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

            # Apply TPMS or Strut Lattice if requested
            scaffold_crop = mask_crop.copy()
            if self.pattern and self.pattern != "None":
                arch_label = "Strut Lattice" if is_strut_pattern(self.pattern) else "TPMS"
                self.progress.emit(
                    40,
                    f"Generating {arch_label} ({self.pattern}, Pore: {self.pore_size_um} µm)…"
                )
                try:
                    scaffold_crop = apply_scaffold_to_crop(
                        mask_crop, self.pattern, self.tpms_type,
                        self.pore_size_um, self.level_set,
                        self.thickness, strut_diam_um=self.strut_diam_um,
                        spacing=self.spacing, invert=self.invert,
                    )
                except Exception as e:
                    print(f"Scaffold pattern error (continuing with plain mask): {e}",
                          file=sys.stderr)
                    scaffold_crop = mask_crop.copy()

            # --- Build full-volume scaffold mask ---
            full_scaffold_mask = np.zeros_like(self.volume, dtype=np.uint8)
            full_scaffold_mask[lo:hi + 1] = scaffold_crop
            if not np.any(full_scaffold_mask):
                raise RuntimeError(
                    "Resulting mask is empty — check HU range, crop, or pore/strut dimensions vs. voxel resolution"
                )

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
            bone_pv_mesh = None
            self.progress.emit(75, "Building host bone mesh for overlay…")
            try:
                host_bone_mask = np.zeros_like(self.volume, dtype=np.uint8)

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
#  SLICE VIEW WIDGET  (custom QWidget for 2D CT viewport with ±90° rotation)
# ══════════════════════════════════════════════════════════════════════════════

class SliceViewWidget(QWidget):
    """
    Renders a single orthogonal CT slice with:
      - 2D planar reorientation in ±90° increments (0°, 90°, 180°, 270°)
      - Bidirectional coordinate mapping for crosshairs & click navigation
      - Dynamic anatomical orientation labels that rotate with the view
      - Window/Level contrast (right-click drag)
      - Coloured crosshair lines synchronised with other views
      - Semi-transparent mask overlay
      - Mouse-wheel slice scrolling
      - Click-to-navigate (left button)
    """
    # Signals
    slice_changed     = pyqtSignal(str, int)        # (plane, new_index)
    click_navigated   = pyqtSignal(str, int, int)   # (plane, unrot_row, unrot_col)
    wl_changed_sig    = pyqtSignal(int, int)        # (wl, ww) broadcast
    rotation_changed  = pyqtSignal(str, int)        # (plane, degrees_cw)

    def __init__(self, plane, border_color, orientation_labels, parent=None):
        super().__init__(parent)
        self.plane = plane
        self.border_color = border_color
        self.base_ori_labels = dict(orientation_labels)
        self.ori_labels = dict(orientation_labels)

        # Rotation state: clockwise 90° steps in {0, 1, 2, 3}
        self.rotation_k = 0

        # Volume references (set by MainWindow)
        self.volume_data = None   # int16  (Z, Y, X)
        self.mask_data   = None   # uint8  (Z, Y, X)

        # Slice state
        self.slice_index = 0
        self.max_index   = 0

        # Canonical unrotated crosshair positions (row, col in unrotated 2D slice)
        self.cross_h = 0   # unrotated row
        self.cross_v = 0   # unrotated col

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

        # Cached QImage for the current (rotated) display
        self._qimage = None
        # Unrotated slice dimensions (H_unrot, W_unrot)
        self._unrot_dims = (1, 1)
        # Bounding rect where image is painted (for coordinate mapping)
        self._img_rect = (0, 0, 1, 1)   # (x, y, w, h)

        self.setMinimumSize(160, 160)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)

    # ── rotation API ──────────────────────────────────────────────────────

    def rotate_step(self, delta_k):
        """
        Rotate the 2D view by delta_k * 90° clockwise.
        delta_k = +1 for +90° (CW), delta_k = -1 for -90° (CCW).
        """
        self.set_rotation_k((self.rotation_k + delta_k) % 4)

    def set_rotation_k(self, k):
        """Set clockwise rotation index k in {0, 1, 2, 3}."""
        k = int(k) % 4
        if k != self.rotation_k:
            self.rotation_k = k
            self._update_rotated_labels()
            self._rebuild_image()
            self.rotation_changed.emit(self.plane, self.rotation_degrees)

    @property
    def rotation_degrees(self):
        return self.rotation_k * 90

    def _update_rotated_labels(self):
        """Rotate cardinal orientation labels synchronously with rotation_k."""
        dirs = ['top', 'right', 'bottom', 'left']
        k = self.rotation_k
        new_labels = {}
        for i, d in enumerate(dirs):
            src_dir = dirs[(i - k) % 4]
            if src_dir in self.base_ori_labels:
                new_labels[d] = self.base_ori_labels[src_dir]
        self.ori_labels = new_labels

    def _unrot_to_rot_coords(self, r, c):
        """Map canonical unrotated (row, col) to rotated display (row_rot, col_rot)."""
        H, W = self._unrot_dims
        k = self.rotation_k
        if k == 0:
            return r, c
        elif k == 1:  # 90° CW
            return c, max(0, H - 1 - r)
        elif k == 2:  # 180°
            return max(0, H - 1 - r), max(0, W - 1 - c)
        else:         # 270° CW (90° CCW)
            return max(0, W - 1 - c), r

    def _rot_to_unrot_coords(self, r_rot, c_rot):
        """Inverse map rotated display (row_rot, col_rot) to canonical unrotated (row, col)."""
        H, W = self._unrot_dims
        k = self.rotation_k
        if k == 0:
            r, c = r_rot, c_rot
        elif k == 1:  # 90° CW
            r, c = H - 1 - c_rot, r_rot
        elif k == 2:  # 180°
            r, c = H - 1 - r_rot, W - 1 - c_rot
        else:         # 270° CW (90° CCW)
            r, c = c_rot, W - 1 - r_rot
        return int(np.clip(r, 0, max(0, H - 1))), int(np.clip(c, 0, max(0, W - 1)))

    # ── data setters ──────────────────────────────────────────────────────

    def set_volume(self, vol):
        self.volume_data = vol
        if self.mask_data is not None and (vol is None or self.mask_data.shape != vol.shape):
            self.mask_data = None
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
        """Return the unrotated 2-D HU slice for the current plane and index."""
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
        """Recompute the RGBA QImage from current slice, rotation, windowing, and mask."""
        raw = self._extract_slice()
        if raw is None:
            self._qimage = None
            self.update()
            return

        self._unrot_dims = (raw.shape[0], raw.shape[1])

        if self.rotation_k != 0:
            raw = np.rot90(raw, k=-self.rotation_k)

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
            if self.rotation_k != 0:
                msl = np.rot90(msl, k=-self.rotation_k)
            if msl.shape == gray.shape:
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
        """Return (img_h, img_w) of the current (rotated) slice image."""
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
        """Map a widget pixel coordinate to rotated display (col_rot, row_rot)."""
        dx, dy, dw, dh = self._img_rect
        ih, iw = self._image_dims()
        col = (wx - dx) / max(dw, 1) * iw
        row = (wy - dy) / max(dh, 1) * ih
        return int(np.clip(col, 0, iw - 1)), int(np.clip(row, 0, ih - 1))

    def _image_to_widget(self, col_rot, row_rot):
        """Map rotated display (col_rot, row_rot) to widget pixel coordinate."""
        dx, dy, dw, dh = self._img_rect
        ih, iw = self._image_dims()
        wx = dx + col_rot / max(iw, 1) * dw
        wy = dy + row_rot / max(ih, 1) * dh
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

        # ── crosshairs (transformed by rotation_k) ──
        rot_row, rot_col = self._unrot_to_rot_coords(self.cross_h, self.cross_v)
        if self.rotation_k % 2 == 0:
            pen_row = QPen(QColor(0, 255, 0, 160), 1, Qt.DashLine)
            pen_col = QPen(QColor(0, 200, 255, 160), 1, Qt.DashLine)
        else:
            pen_row = QPen(QColor(0, 200, 255, 160), 1, Qt.DashLine)
            pen_col = QPen(QColor(0, 255, 0, 160), 1, Qt.DashLine)

        _, wy = self._image_to_widget(0, rot_row)
        p.setPen(pen_row)
        p.drawLine(dx, wy, dx + dw, wy)

        wx, _ = self._image_to_widget(rot_col, 0)
        p.setPen(pen_col)
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
        info = f"WL:{self.wl}  WW:{self.ww}  Rot:{self.rotation_degrees}°"
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
            col_rot, row_rot = self._widget_to_image(ev.x(), ev.y())
            unrot_row, unrot_col = self._rot_to_unrot_coords(row_rot, col_rot)
            self.click_navigated.emit(self.plane, unrot_row, unrot_col)
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
#  VIEWPORT PANEL  (header with ±90° rotation + slice view + slider)
# ══════════════════════════════════════════════════════════════════════════════

class SliceViewportPanel(QFrame):
    """
    Composite widget that pairs a SliceViewWidget with:
      - A compact header toolbar containing plane label, rotation badge,
        and ↺ -90° / ↻ +90° reorientation buttons
      - A vertical QSlider along its right edge for slice scrolling
    """
    def __init__(self, slice_view, parent=None):
        super().__init__(parent)
        self.slice_view = slice_view

        header = QWidget()
        header.setFixedHeight(26)
        header.setStyleSheet("background-color: #222; border-bottom: 1px solid #444;")
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(6, 2, 6, 2)
        h_lay.setSpacing(4)

        self.lbl_title = QLabel(f"{self.slice_view.plane.capitalize()}")
        self.lbl_title.setStyleSheet(
            f"color: {self.slice_view.border_color.name()}; font-weight: bold; font-size: 11px;"
        )
        h_lay.addWidget(self.lbl_title)
        h_lay.addStretch(1)

        self.lbl_rot = QLabel("0°")
        self.lbl_rot.setStyleSheet("color: #aaa; font-family: Consolas; font-size: 10px;")
        h_lay.addWidget(self.lbl_rot)

        self.btn_rot_ccw = QPushButton("↺ -90°")
        self.btn_rot_ccw.setToolTip("Rotate 2D view -90° (Counter-Clockwise)")
        self.btn_rot_ccw.setFixedHeight(20)
        self.btn_rot_ccw.setStyleSheet("font-size: 10px; padding: 1px 6px;")
        self.btn_rot_ccw.clicked.connect(lambda: self.slice_view.rotate_step(-1))
        h_lay.addWidget(self.btn_rot_ccw)

        self.btn_rot_cw = QPushButton("↻ +90°")
        self.btn_rot_cw.setToolTip("Rotate 2D view +90° (Clockwise)")
        self.btn_rot_cw.setFixedHeight(20)
        self.btn_rot_cw.setStyleSheet("font-size: 10px; padding: 1px 6px;")
        self.btn_rot_cw.clicked.connect(lambda: self.slice_view.rotate_step(1))
        h_lay.addWidget(self.btn_rot_cw)

        self.slider = QSlider(Qt.Vertical)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setInvertedAppearance(True)
        self.slider.setFixedWidth(18)
        self.slider.setStyleSheet("""
            QSlider::groove:vertical { background: #444; width: 6px; }
            QSlider::handle:vertical {
                background: #aaa; height: 14px; margin: 0 -4px;
                border-radius: 3px;
            }
        """)

        body_lay = QHBoxLayout()
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(0)
        body_lay.addWidget(self.slice_view, 1)
        body_lay.addWidget(self.slider, 0)

        main_lay = QVBoxLayout(self)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)
        main_lay.addWidget(header, 0)
        main_lay.addLayout(body_lay, 1)

        self.slider.valueChanged.connect(self._on_slider)
        self.slice_view.slice_changed.connect(self._on_view_scroll)
        self.slice_view.rotation_changed.connect(self._on_rotation_changed)

    def _on_slider(self, val):
        self.slice_view.set_slice(val)
        self.slice_view.slice_changed.emit(self.slice_view.plane, val)

    def _on_view_scroll(self, _plane, idx):
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)

    def _on_rotation_changed(self, _plane, deg):
        self.lbl_rot.setText(f"{deg}°")

    def update_range(self, max_idx):
        self.slider.setMaximum(max_idx)


class ThreeDViewportPanel(QFrame):
    """Wraps a pyvistaqt.QtInteractor in a styled frame."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("border: 3px solid " + COLOR_3D.name() + ";")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)

        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        self.scaffold_mesh = None
        self.bone_mesh = None
        self.scaffold_actor = None
        self.bone_actor = None
        self.overlay_enabled = False
        self.bone_alpha = DEFAULT_BONE_ALPHA

        self._init_lighting_and_rendering()

    def _init_lighting_and_rendering(self):
        try:
            self.plotter.set_background(color="#1a1c22", top="#0a0c10")
        except Exception:
            self.plotter.set_background("#1a1a1a")

        try:
            self.plotter.enable_eye_dome_lighting()
        except Exception:
            pass

        try:
            self.plotter.enable_depth_peeling(
                number_of_peels=4, occlusion_ratio=0.0
            )
        except Exception:
            pass

        self.plotter.remove_all_lights()

        key_light = pv.Light(
            position=(1.0, 1.0, 2.0),
            focal_point=(0.0, 0.0, 0.0),
            color="white",
            intensity=0.85,
        )
        key_light.positional = False
        self.plotter.add_light(key_light)

        fill_light = pv.Light(
            position=(-2.0, -1.0, -0.5),
            focal_point=(0.0, 0.0, 0.0),
            color="#B0C4DE",
            intensity=0.35,
        )
        fill_light.positional = False
        self.plotter.add_light(fill_light)

        rim_light = pv.Light(
            position=(0.0, 2.0, -2.0),
            focal_point=(0.0, 0.0, 0.0),
            color="white",
            intensity=0.50,
        )
        rim_light.positional = False
        self.plotter.add_light(rim_light)

        self.plotter.add_axes()

    def display_mesh(self, pv_mesh, color=None):
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
        self.scaffold_mesh = scaffold_mesh
        self.bone_mesh = bone_mesh

        self.plotter.clear()
        self.scaffold_actor = None
        self.bone_actor = None

        if scaffold_mesh is not None and scaffold_mesh.n_points > 0:
            self.scaffold_actor = self.plotter.add_mesh(
                scaffold_mesh, color=SCAFFOLD_COLOR,
                smooth_shading=True,
                ambient=0.20, diffuse=0.80,
                specular=0.35, specular_power=25,
                name="scaffold",
            )

        if bone_mesh is not None and bone_mesh.n_points > 0:
            self.bone_actor = self.plotter.add_mesh(
                bone_mesh, color=OVERLAY_BONE_COLOR,
                smooth_shading=True,
                ambient=0.15, diffuse=0.75,
                specular=0.20, specular_power=15,
                opacity=self.bone_alpha if self.overlay_enabled else 0.0,
                name="overlay_bone",
            )
            if not self.overlay_enabled and self.bone_actor is not None:
                try:
                    self.bone_actor.SetVisibility(False)
                except Exception:
                    pass

        self.plotter.add_axes()
        self.plotter.reset_camera()

    def set_overlay_enabled(self, enabled):
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
        self.bone_alpha = alpha
        if self.bone_actor is not None and self.overlay_enabled:
            try:
                self.bone_actor.GetProperty().SetOpacity(alpha)
                self.plotter.render()
            except Exception:
                pass

    def reset_camera(self):
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
      2. 2D Viewport Reorientation (±90°)
      3. Threshold & Crop
      4. Scaffold Pattern (TPMS + Strut Lattices, Physical Pore Size 50–1500 µm)
      5. Mesh Operations & Export
      6. 3D View & Anatomical Overlay
    """
    sig_select_dicom    = pyqtSignal()
    sig_select_nrrd     = pyqtSignal()
    sig_load_volume     = pyqtSignal()
    sig_rotate_plane    = pyqtSignal(str, int)   # (plane, delta_k)
    sig_reset_rotations = pyqtSignal()
    sig_preview_thresh  = pyqtSignal()
    sig_segment         = pyqtSignal()
    sig_smooth_mesh     = pyqtSignal()
    sig_export_stl      = pyqtSignal()
    sig_overlay_toggled = pyqtSignal(bool)
    sig_alpha_changed   = pyqtSignal(float)
    sig_reset_camera    = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(310)
        self.setMaximumWidth(410)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._current_spacing = (1.0, 1.0, 1.0)

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

        # ── 2. 2D Viewport Orientation (±90°) ────────────────────────────
        grp_orient = CollapsibleGroupBox("2D Viewport Reorientation")
        lay_or = QVBoxLayout()

        self.lbl_rot_status = {}
        for plane_key, label_txt, col_hex in (
            ("coronal",  "Coronal:",  "#00c800"),
            ("axial",    "Axial:",    "#dc2828"),
            ("sagittal", "Sagittal:", "#3c78ff"),
        ):
            row_r = QHBoxLayout()
            lbl_p = QLabel(label_txt)
            lbl_p.setStyleSheet(f"color: {col_hex}; font-weight: bold;")
            lbl_p.setFixedWidth(65)
            row_r.addWidget(lbl_p)

            btn_ccw = QPushButton("↺ -90°")
            btn_cw  = QPushButton("↻ +90°")
            btn_ccw.setFixedHeight(24)
            btn_cw.setFixedHeight(24)
            btn_ccw.clicked.connect(
                lambda _, p=plane_key: self.sig_rotate_plane.emit(p, -1)
            )
            btn_cw.clicked.connect(
                lambda _, p=plane_key: self.sig_rotate_plane.emit(p, 1)
            )
            row_r.addWidget(btn_ccw, 1)
            row_r.addWidget(btn_cw, 1)

            lbl_deg = QLabel("0°")
            lbl_deg.setFixedWidth(36)
            lbl_deg.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl_deg.setStyleSheet("color: #aaa; font-family: Consolas;")
            row_r.addWidget(lbl_deg)
            self.lbl_rot_status[plane_key] = lbl_deg
            lay_or.addLayout(row_r)

        self.btn_reset_rot = QPushButton("Reset All 2D Orientations (0°)")
        self.btn_reset_rot.clicked.connect(self.sig_reset_rotations)
        lay_or.addWidget(self.btn_reset_rot)

        grp_orient.setLayout(lay_or)
        main_lay.addWidget(grp_orient)

        # ── 3. Threshold & Crop ───────────────────────────────────────────
        grp_thresh = CollapsibleGroupBox("Threshold && Crop")
        lay_t = QVBoxLayout()

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

        # ── 4. Scaffold Pattern (TPMS + Strut Lattices) ───────────────────
        grp_tpms = CollapsibleGroupBox("Scaffold Architecture (TPMS && Strut)")
        lay_p = QVBoxLayout()

        # Pattern selector (Default set to "None")
        row_pat = QHBoxLayout()
        row_pat.addWidget(QLabel("Pattern:"))
        self.combo_pattern = QComboBox()
        self.combo_pattern.addItems(ALL_SCAFFOLD_PATTERNS)
        self.combo_pattern.setCurrentText("None")
        row_pat.addWidget(self.combo_pattern, 1)
        lay_p.addLayout(row_pat)

        self.lbl_arch_badge = QLabel("Architecture: Solid Bone (No Pattern)")
        self.lbl_arch_badge.setStyleSheet("color: #aaa; font-size: 10px; font-style: italic;")
        lay_p.addWidget(self.lbl_arch_badge)

        # Invert
        self.chk_invert = QCheckBox("Invert Solid / Void Phase")
        lay_p.addWidget(self.chk_invert)

        # Target Pore Size (µm) replacing dimensionless Cell Count
        lay_p.addWidget(QLabel("Target Pore Size (50 – 1500 µm):"))
        row_pore = QHBoxLayout()
        self.spin_pore_um = QSpinBox()
        self.spin_pore_um.setRange(MIN_PORE_UM, MAX_PORE_UM)
        self.spin_pore_um.setSingleStep(STEP_PORE_UM)
        self.spin_pore_um.setSuffix(" µm")
        self.spin_pore_um.setValue(DEFAULT_PORE_UM)
        row_pore.addWidget(self.spin_pore_um)

        self.slider_pore_um = QSlider(Qt.Horizontal)
        self.slider_pore_um.setRange(MIN_PORE_UM, MAX_PORE_UM)
        self.slider_pore_um.setSingleStep(STEP_PORE_UM)
        self.slider_pore_um.setPageStep(100)
        self.slider_pore_um.setValue(DEFAULT_PORE_UM)
        row_pore.addWidget(self.slider_pore_um, 1)
        lay_p.addLayout(row_pore)

        self.lbl_pore_mm = QLabel("Metric Pore Size: 500 µm (0.500 mm)")
        self.lbl_pore_mm.setStyleSheet("color: #8cd; font-size: 10px; font-family: Consolas;")
        lay_p.addWidget(self.lbl_pore_mm)

        self.spin_pore_um.valueChanged.connect(self._on_pore_changed)
        self.slider_pore_um.valueChanged.connect(self.spin_pore_um.setValue)

        # ── TPMS-specific sub-panel ──
        self.tpms_params_widget = QWidget()
        tpms_sub = QVBoxLayout(self.tpms_params_widget)
        tpms_sub.setContentsMargins(0, 2, 0, 2)
        tpms_sub.setSpacing(4)

        row_type = QHBoxLayout()
        row_type.addWidget(QLabel("TPMS Type:"))
        self.combo_type = QComboBox()
        self.combo_type.addItems(["Skeletal", "Shell"])
        row_type.addWidget(self.combo_type, 1)
        tpms_sub.addLayout(row_type)

        row_level = QHBoxLayout()
        self.lbl_level_title = QLabel("Level-Set:")
        row_level.addWidget(self.lbl_level_title)
        self.slider_level = QSlider(Qt.Horizontal)
        self.slider_level.setRange(-200, 200)
        self.slider_level.setValue(0)
        row_level.addWidget(self.slider_level, 1)
        self.lbl_level_val = QLabel("0.00")
        self.lbl_level_val.setFixedWidth(40)
        row_level.addWidget(self.lbl_level_val)
        tpms_sub.addLayout(row_level)
        self.slider_level.valueChanged.connect(self._on_level_changed)

        row_thick = QHBoxLayout()
        self.lbl_thick_title = QLabel("Thickness:")
        row_thick.addWidget(self.lbl_thick_title)
        self.slider_thick = QSlider(Qt.Horizontal)
        self.slider_thick.setRange(1, 50)
        self.slider_thick.setValue(10)
        row_thick.addWidget(self.slider_thick, 1)
        self.lbl_thick_val = QLabel("0.10")
        self.lbl_thick_val.setFixedWidth(40)
        row_thick.addWidget(self.lbl_thick_val)
        tpms_sub.addLayout(row_thick)
        self.slider_thick.valueChanged.connect(self._on_thick_changed)

        lay_p.addWidget(self.tpms_params_widget)

        # ── Strut-Lattice specific sub-panel ──
        self.strut_params_widget = QWidget()
        strut_sub = QVBoxLayout(self.strut_params_widget)
        strut_sub.setContentsMargins(0, 2, 0, 2)
        strut_sub.setSpacing(4)

        strut_sub.addWidget(QLabel("Strut Diameter / Thickness (50 – 800 µm):"))
        row_strut = QHBoxLayout()
        self.spin_strut_um = QSpinBox()
        self.spin_strut_um.setRange(MIN_STRUT_UM, MAX_STRUT_UM)
        self.spin_strut_um.setSingleStep(STEP_STRUT_UM)
        self.spin_strut_um.setSuffix(" µm")
        self.spin_strut_um.setValue(DEFAULT_STRUT_UM)
        row_strut.addWidget(self.spin_strut_um)

        self.slider_strut_um = QSlider(Qt.Horizontal)
        self.slider_strut_um.setRange(MIN_STRUT_UM, MAX_STRUT_UM)
        self.slider_strut_um.setSingleStep(STEP_STRUT_UM)
        self.slider_strut_um.setPageStep(50)
        self.slider_strut_um.setValue(DEFAULT_STRUT_UM)
        row_strut.addWidget(self.slider_strut_um, 1)
        strut_sub.addLayout(row_strut)

        self.lbl_strut_mm = QLabel("Metric Strut Diameter: 200 µm (0.200 mm)")
        self.lbl_strut_mm.setStyleSheet("color: #8cd; font-size: 10px; font-family: Consolas;")
        strut_sub.addWidget(self.lbl_strut_mm)

        self.spin_strut_um.valueChanged.connect(self._on_strut_changed)
        self.slider_strut_um.valueChanged.connect(self.spin_strut_um.setValue)

        lay_p.addWidget(self.strut_params_widget)

        # Physical unit-cell period & Nyquist resolution readout
        self.lbl_pore_info = QLabel("")
        self.lbl_pore_info.setWordWrap(True)
        self.lbl_pore_info.setStyleSheet(
            "color: #bbb; font-size: 10px; font-family: Consolas; "
            "background: #222; padding: 4px; border-radius: 3px;"
        )
        lay_p.addWidget(self.lbl_pore_info)

        # Wire dynamic UI updates
        self.combo_pattern.currentTextChanged.connect(self._on_pattern_changed)
        self.combo_type.currentTextChanged.connect(self._on_type_changed)
        self._on_type_changed(self.combo_type.currentText())
        self._on_pattern_changed(self.combo_pattern.currentText())

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

        # ── 5. Mesh Operations & Export ───────────────────────────────────
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

        # ── 6. 3D View & Anatomical Overlay ──────────────────────────────
        grp_overlay = CollapsibleGroupBox("3D View && Anatomical Overlay")
        lay_o = QVBoxLayout()

        self.chk_overlay = QCheckBox("Overlay Host Bone")
        self.chk_overlay.setChecked(False)
        self.chk_overlay.setEnabled(False)
        self.chk_overlay.toggled.connect(self._on_overlay_toggled)
        lay_o.addWidget(self.chk_overlay)

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

        self.btn_reset_cam = QPushButton("Reset 3D Camera")
        self.btn_reset_cam.clicked.connect(self.sig_reset_camera)
        lay_o.addWidget(self.btn_reset_cam)

        grp_overlay.setLayout(lay_o)
        main_lay.addWidget(grp_overlay)

        # Spacer
        main_lay.addStretch(1)

        self.setWidget(container)

    # ── helpers ───────────────────────────────────────────────────────────

    def set_voxel_spacing(self, spacing_zyx):
        """Store current volume voxel spacing (sz, sy, sx) in mm and refresh diagnostics."""
        self._current_spacing = tuple(float(s) for s in spacing_zyx)
        self._update_pore_diagnostics()

    def update_rotation_label(self, plane, degrees):
        if plane in self.lbl_rot_status:
            self.lbl_rot_status[plane].setText(f"{degrees}°")

    def _on_pattern_changed(self, pat):
        strut_mode = is_strut_pattern(pat)
        none_mode = (pat == "None")

        self.tpms_params_widget.setVisible(not strut_mode and not none_mode)
        self.strut_params_widget.setVisible(strut_mode)

        if none_mode:
            self.lbl_arch_badge.setText("Architecture: Solid Bone (No Pattern)")
        elif strut_mode:
            desc_map = {
                "Cubic": "Strut Lattice — Simple Cubic Orthogonal Frame",
                "BCC":   "Strut Lattice — Body-Centered Cubic (8-Corner Diagonals)",
                "BCCZ":  "Strut Lattice — BCC + Load-Bearing Z-Axis Columns",
                "FCC":   "Strut Lattice — Face-Centered Cubic Diagonals",
            }
            self.lbl_arch_badge.setText(desc_map.get(pat, "Strut-Based Lattice"))
        else:
            self.lbl_arch_badge.setText(f"TPMS Implicit Surface — {pat}")

        self._update_pore_diagnostics()

    def _on_pore_changed(self, val):
        self.slider_pore_um.blockSignals(True)
        self.slider_pore_um.setValue(val)
        self.slider_pore_um.blockSignals(False)
        self.lbl_pore_mm.setText(f"Metric Pore Size: {val} µm ({val / 1000.0:.3f} mm)")
        self._update_pore_diagnostics()

    def _on_strut_changed(self, val):
        self.slider_strut_um.blockSignals(True)
        self.slider_strut_um.setValue(val)
        self.slider_strut_um.blockSignals(False)
        self.lbl_strut_mm.setText(f"Metric Strut Diameter: {val} µm ({val / 1000.0:.3f} mm)")
        self._update_pore_diagnostics()

    def _on_level_changed(self, v):
        self.lbl_level_val.setText(f"{v / 100:.2f}")
        self._update_pore_diagnostics()

    def _on_thick_changed(self, v):
        self.lbl_thick_val.setText(f"{v / 100:.2f}")
        self._update_pore_diagnostics()

    def _update_pore_diagnostics(self):
        pat = self.combo_pattern.currentText()
        pore_um = self.spin_pore_um.value()
        pore_mm = pore_um / 1000.0
        strut_um = self.spin_strut_um.value()
        strut_mm = strut_um / 1000.0
        max_sp = max(self._current_spacing)
        nyquist_mm = 2.0 * max_sp

        if pat == "None":
            self.lbl_pore_info.setText(
                "Pattern: None (Solid segmented bone without porous cutouts)"
            )
            self.lbl_pore_info.setStyleSheet(
                "color: #aaa; font-size: 10px; font-family: Consolas; "
                "background: #222; padding: 4px; border-radius: 3px;"
            )
            return

        L_mm = compute_unit_cell_period_mm(
            pat, self.tpms_type, pore_um,
            level_set=self.tpms_level, thickness=self.tpms_thickness,
            strut_diam_um=strut_um,
        )

        min_feature_mm = min(pore_mm, strut_mm) if is_strut_pattern(pat) else pore_mm
        if min_feature_mm < nyquist_mm:
            warn_txt = (
                f"⚠ Resolution Warning: Feature size ({min_feature_mm:.3f} mm) is below "
                f"2× max voxel spacing ({nyquist_mm:.3f} mm). "
                f"CT grid aliasing or strut disconnection may occur."
            )
            color_hex = "#f94"
        else:
            warn_txt = (
                f"✓ Nyquist Check OK: Feature ({min_feature_mm:.3f} mm) ≥ "
                f"2× max spacing ({nyquist_mm:.3f} mm)."
            )
            color_hex = "#7d9"

        extra_line = ""
        if is_strut_pattern(pat):
            extra_line = f"Strut Ø: {strut_mm:.3f} mm  |  Pore Ø: {pore_mm:.3f} mm\n"

        self.lbl_pore_info.setText(
            f"Unit Cell Period L: {L_mm:.3f} mm ({L_mm * 1000:.0f} µm)\n"
            f"{extra_line}{warn_txt}"
        )
        self.lbl_pore_info.setStyleSheet(
            f"color: {color_hex}; font-size: 10px; font-family: Consolas; "
            "background: #222; padding: 4px; border-radius: 3px;"
        )

    def _apply_preset(self, hu_lo, hu_hi, wl, ww):
        self.spin_hu_min.setValue(hu_lo)
        self.spin_hu_max.setValue(hu_hi)
        self.wl_preset = (wl, ww)
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
        self._update_pore_diagnostics()

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

    # Accessor properties for reading current scaffold settings
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
    def pore_size_um(self):
        return self.spin_pore_um.value()

    @property
    def strut_diam_um(self):
        return self.spin_strut_um.value()

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
        self.sig_overlay_toggled.emit(checked)

    def _on_alpha_slider(self, val):
        self.lbl_alpha_val.setText(f"{val}%")
        self.sig_alpha_changed.emit(val / 100.0)

    def enable_overlay_controls(self, enabled):
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
    quad-viewport medical imaging workstation (Version B).
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Andamyo v3.B — CT Bone Segmentation, TPMS & Strut-Lattice Scaffold Tool"
        )
        self.resize(1420, 920)

        # ── application state ─────────────────────────────────────────────
        self.volume_data    = None   # int16 (Z, Y, X)
        self.segmented_data = None   # uint8 (Z, Y, X)
        self.image_spacing  = (1.0, 1.0, 1.0)  # (Z, Y, X) in mm
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

        self.views_by_plane = {
            'coronal':  self.view_coronal,
            'axial':    self.view_axial,
            'sagittal': self.view_sagittal,
        }

        self.vp_coronal  = SliceViewportPanel(self.view_coronal)
        self.vp_axial    = SliceViewportPanel(self.view_axial)
        self.vp_sagittal = SliceViewportPanel(self.view_sagittal)

        self.vp_3d = ThreeDViewportPanel()

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

        for view in (self.view_coronal, self.view_axial, self.view_sagittal):
            view.slice_changed.connect(self._on_slice_changed)
            view.click_navigated.connect(self._on_click_navigated)
            view.wl_changed_sig.connect(self._on_wl_changed_from_view)
            view.rotation_changed.connect(self._on_view_rotation_changed)

    def _build_control_dock(self):
        """Create the right-side dockable control panel."""
        self.ctrl = ControlPanel()

        dock = QDockWidget("Controls", self)
        dock.setWidget(self.ctrl)
        dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self.ctrl.sig_select_dicom.connect(self._select_dicom_folder)
        self.ctrl.sig_select_nrrd.connect(self._select_nrrd_file)
        self.ctrl.sig_load_volume.connect(self._load_volume)
        self.ctrl.sig_rotate_plane.connect(self._on_rotate_plane_requested)
        self.ctrl.sig_reset_rotations.connect(self._on_reset_rotations)
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
    #  2D VIEWPORT REORIENTATION HANDLERS
    # ══════════════════════════════════════════════════════════════════════

    def _on_rotate_plane_requested(self, plane, delta_k):
        view = self.views_by_plane.get(plane)
        if view is not None:
            view.rotate_step(delta_k)

    def _on_reset_rotations(self):
        for view in self.views_by_plane.values():
            view.set_rotation_k(0)

    def _on_view_rotation_changed(self, plane, degrees):
        if hasattr(self, 'ctrl') and self.ctrl is not None:
            self.ctrl.update_rotation_label(plane, degrees)

    # ══════════════════════════════════════════════════════════════════════
    #  MPR CROSSHAIR SYNCHRONISATION
    # ══════════════════════════════════════════════════════════════════════

    def _sync_all_views(self):
        """Push current (cur_z, cur_y, cur_x) to all slice views."""
        if self.volume_data is None:
            return
        z, y, x = self.cur_z, self.cur_y, self.cur_x

        self.view_axial.set_slice(z)
        self.view_axial.set_crosshairs(y, x)
        self.vp_axial.slider.blockSignals(True)
        self.vp_axial.slider.setValue(z)
        self.vp_axial.slider.blockSignals(False)

        self.view_coronal.set_slice(y)
        self.view_coronal.set_crosshairs(z, x)
        self.vp_coronal.slider.blockSignals(True)
        self.vp_coronal.slider.setValue(y)
        self.vp_coronal.slider.blockSignals(False)

        self.view_sagittal.set_slice(x)
        self.view_sagittal.set_crosshairs(z, y)
        self.vp_sagittal.slider.blockSignals(True)
        self.vp_sagittal.slider.setValue(x)
        self.vp_sagittal.slider.blockSignals(False)

    def _on_slice_changed(self, plane, idx):
        if plane == 'axial':
            self.cur_z = idx
        elif plane == 'coronal':
            self.cur_y = idx
        elif plane == 'sagittal':
            self.cur_x = idx
        self._sync_all_views()

    def _on_click_navigated(self, plane, row, col):
        if plane == 'axial':
            self.cur_y = row
            self.cur_x = col
        elif plane == 'coronal':
            self.cur_z = row
            self.cur_x = col
        elif plane == 'sagittal':
            self.cur_z = row
            self.cur_y = col
        self._sync_all_views()

    def _on_wl_changed_from_view(self, wl, ww):
        for v in (self.view_axial, self.view_coronal, self.view_sagittal):
            v.set_wl_ww(wl, ww)

    def apply_wl_preset(self, wl, ww):
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
        self.ctrl.set_voxel_spacing(spacing)

        self.cur_z = shape[0] // 2
        self.cur_y = shape[1] // 2
        self.cur_x = shape[2] // 2

        # Push volume to views (clear old masks first to prevent dimension mismatch)
        for v in (self.view_axial, self.view_coronal, self.view_sagittal):
            v.set_mask(None)
            v.set_volume(vol)

        self.vp_axial.update_range(shape[0] - 1)
        self.vp_coronal.update_range(shape[1] - 1)
        self.vp_sagittal.update_range(shape[2] - 1)

        self.ctrl.update_crop_range(shape[0] - 1)

        self._sync_all_views()

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

        full_mask = np.zeros_like(self.volume_data, dtype=np.uint8)
        raw = (self.volume_data[lo:hi+1] >= hu_min) & \
              (self.volume_data[lo:hi+1] <= hu_max)
        full_mask[lo:hi+1] = raw.astype(np.uint8)

        self.segmented_data = full_mask

        for v in (self.view_axial, self.view_coronal, self.view_sagittal):
            v.set_mask(full_mask)

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
            volume        = self.volume_data,
            spacing       = self.image_spacing,
            hu_min        = self.ctrl.hu_min,
            hu_max        = self.ctrl.hu_max,
            crop_lo       = self.ctrl.crop_lo,
            crop_hi       = self.ctrl.crop_hi,
            pattern       = self.ctrl.tpms_pattern,
            tpms_type     = self.ctrl.tpms_type,
            pore_size_um  = self.ctrl.pore_size_um,
            level_set     = self.ctrl.tpms_level,
            thickness     = self.ctrl.tpms_thickness,
            strut_diam_um = self.ctrl.strut_diam_um,
            invert        = self.ctrl.tpms_invert,
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
        self.current_pv_mesh = scaffold_mesh
        self.current_scaffold_mesh = scaffold_mesh
        self.current_bone_mesh = bone_mesh

        for v in (self.view_axial, self.view_coronal, self.view_sagittal):
            v.set_mask(full_mask)

        self.vp_3d.set_meshes(scaffold_mesh, bone_mesh)
        self._update_mesh_stats(scaffold_mesh)

        has_bone = bone_mesh is not None and bone_mesh.n_points > 0
        self.ctrl.enable_overlay_controls(has_bone)

        pat_info = self.ctrl.tpms_pattern
        if is_strut_pattern(pat_info):
            pat_info += (
                f" (Pore: {self.ctrl.pore_size_um} µm, "
                f"Strut: {self.ctrl.strut_diam_um} µm)"
            )
        elif pat_info != "None":
            pat_info += f" (Pore: {self.ctrl.pore_size_um} µm)"

        self.lbl_status.setText(
            f"Segmentation complete  |  Pattern: {pat_info}"
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
        self.vp_3d.set_meshes(pv_mesh, self.current_bone_mesh)
        self._update_mesh_stats(pv_mesh)
        self.lbl_status.setText("Mesh smoothing complete")

    # ══════════════════════════════════════════════════════════════════════
    #  3D OVERLAY CONTROLS
    # ══════════════════════════════════════════════════════════════════════

    def _on_toggle_overlay(self, checked):
        self.vp_3d.set_overlay_enabled(checked)

    def _on_overlay_alpha_changed(self, alpha):
        self.vp_3d.set_bone_alpha(alpha)

    def _on_reset_camera(self):
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
        self.vp_3d.close_plotter()
        event.accept()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

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
