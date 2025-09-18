# CAPSTONE-THESIS

CT Bone Segmentation Tool

A Python application for processing CT scan DICOM files, segmenting bones based on Hounsfield unit (HU) ranges, and exporting the result as a 3D STL file.

## Features

- **DICOM File Loading**: Load CT scan series from DICOM files
- **Interactive HU Range Selection**: Set custom Hounsfield unit ranges for bone segmentation
- **Real-time Preview**: Preview segmentation results on CT slices
- **3D Bone Segmentation**: Process entire 3D volume with morphological operations
- **STL Export**: Export segmented bones as 3D STL files for 3D printing or visualization
- **User-friendly GUI**: Simple graphical interface built with tkinter

## Installation

1. Install Python 3.7 or higher
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python ct_bone_segmentation.py
   ```

2. **Load DICOM Files**:
   - Click "Select DICOM Folder" to choose a folder containing DICOM files
   - Click "Load DICOM Series" to load the CT scan data

3. **Set HU Range**:
   - Adjust the Min HU and Max HU values (typical bone range: 150-3000 HU)
   - Click "Preview Segmentation" to see results on a sample slice

4. **Segment Bones**:
   - Click "Segment Bones" to process the entire 3D volume
   - The application will apply morphological operations to clean up the segmentation

5. **Export STL**:
   - Click "Export STL" to save the segmented bones as a 3D STL file
   - Choose the output location and filename

## Hounsfield Unit Ranges

- **Bone**: 150-3000 HU (default range)
- **Cortical Bone**: 200-3000 HU
- **Trabecular Bone**: 150-400 HU
- **Soft Tissue**: -100 to 100 HU
- **Air**: -1000 to -500 HU

## Technical Details

- Uses pydicom for DICOM file reading
- Employs scikit-image for image processing and morphological operations
- Utilizes scipy's marching cubes algorithm for 3D mesh generation
- Exports STL files using numpy-stl library
- Multi-threaded processing to prevent GUI freezing

## Requirements

- Python 3.7+
- pydicom >= 2.3.0
- numpy >= 1.21.0
- scipy >= 1.7.0
- scikit-image >= 0.18.0
- vtk >= 9.0.0
- numpy-stl >= 2.16.0
- matplotlib >= 3.5.0
- Pillow >= 8.3.0

## Troubleshooting

- **Memory Issues**: Large CT volumes may require significant RAM. Consider processing smaller regions or reducing resolution.
- **DICOM Loading**: Ensure DICOM files are properly formatted and contain pixel data.
- **STL Export**: Very large meshes may take time to process and export.

## License

This project is open source and available under the MIT License.
