# orientation-analysis
Algorithm that carries out 2D and 3D orientation analysis using structure tensors of HiP-CT data.


# Examples
In a terminal window enter the name of the program followed by the value for 
σ (sigma) and ρ (rho) respectively.
A window will appear asking you to select the file you want to process.
``` python

python3 dataProcessing3d.py 1 10

```
To visualise the vector orientation results as an RGB image run the following line of code.
A windows will appear asking you to select the nifti file which contains the vector data.
``` python

python3 niftiToRGB.py

```
To convert the nifti vector results to a VTK file for viewing in Paraview, run the following line of code.
A windows will appear asking you to select the nifti file which contains the vector data.
``` python

python3 niftiToVTK.py

```

# Packages Required
- [structure-tensor](https://github.com/py-novice/structure-tensor)
- [nibabel](https://nipy.org/nibabel/)
- numpy
- matplotlib
- scikit-image
- tqdm
- tkinter
- tiffile
- [mayavi](https://docs.enthought.com/mayavi/mayavi/) (if going to save results as a VTK file as well)



