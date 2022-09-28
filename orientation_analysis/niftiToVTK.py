"""
Function to convert nifti vector file to vtk file for better visualization in python
"""
import sys
import numpy as np
import nibabel as nib
import tkinter
from tkinter import filedialog
import time
from tvtk.api import tvtk, write_data
import math

def colouring(vec):
    # function to assign rgba values to data

    vec = abs(vec)  # negative numbers don't matter for assigning colour
    vec[~np.isfinite(vec)] = np.nan  # Replaces any inf values with nan

    input_shape = vec.shape
    vec = vec.reshape(3, -1)
    tmp = np.empty((6,) + vec.shape[1:], dtype="float32")
    # find red
    np.subtract(vec[2], np.nanmin(vec[2]), out=tmp[0])
    np.subtract(np.nanmax(vec[2]), np.nanmin(vec[2]), out=tmp[1])
    r = np.divide(tmp[0], tmp[1])  # x-axis component is the red channel
    # find blue
    np.subtract(vec[1], np.nanmin(vec[1]), out=tmp[2])
    np.subtract(np.nanmax(vec[1]), np.nanmin(vec[1]), out=tmp[3])
    b = np.divide(tmp[2], tmp[3])  # y-axis component is the blue channel
    # find green
    np.subtract(vec[0], np.nanmin(vec[0]), out=tmp[4])
    np.subtract(np.nanmax(vec[0]), np.nanmin(vec[0]), out=tmp[5])
    g = np.divide(tmp[4], tmp[5])  # z-axis component is the green channel

    rgba = np.empty((input_shape[1], input_shape[2], input_shape[3], 4))
    rgba[..., 0] = r.reshape(input_shape[1:])
    rgba[..., 1] = g.reshape(input_shape[1:])
    rgba[..., 2] = b.reshape(input_shape[1:])

    # rgba *= 255  # normalise data
    rgba = rgba.astype(np.uint8)  # convert to uint8 to reduce array size
    del tmp
    return rgba

if __name__ == '__main__':
    # hides root window
    root = tkinter.Tk()
    root.withdraw()

    # ask user to select Vector data and load data using ui
    filename = filedialog.askopenfile(title='Select vector data file to process')
    if not filename:
        # handles when askopenfile dialog is closed with "cancel".
        sys.exit('No file was selected to be processed')

    # ask user location to save results
    save_filename = filedialog.asksaveasfilename(title='select location and enter name of file to save results')
    root.destroy()
    if save_filename == '':  # asksaveasfile return `none` if dialog closed with "cancel".
        sys.exit('no file name or location was selected to save results')

    # Import nifti file
    print('Importing data...')
    data_info = nib.load(filename.name)
    data = data_info.get_fdata()
    print('Finished importing, now reshaping data...')
    eigVec = data.transpose(3, 0, 1, 2) # re-orders data so it's (3,x,y,z)

    # Assign rgb values to vector data
    rgba = colouring(eigVec)

    # eigenvectors
    vx = eigVec[2]
    vy = eigVec[1]
    vz = eigVec[0]
    dim = vx.shape

    # rgb data
    val_major = rgba[..., 0]  # red channel
    val_minor1 = rgba[..., 1]  # green channel
    val_minor2 = rgba[..., 2]  # blue channel

    # generates grid and arranges into correct format.
    x, y, z = np.mgrid[0:dim[0], 0:dim[1], 0:dim[2]]
    pts = np.empty(z.shape + (3,), dtype=np.float32)
    pts[..., 0] = x
    pts[..., 1] = y
    pts[..., 2] = z
    # pts = pts.transpose(3, 0, 1, 2).copy()  # re-orders matrix so x first, y next and z last.

    # arranges vector field data into correct format
    vectors = np.empty(z.shape + (3,), dtype=np.float32)
    vectors[..., 0] = vx
    vectors[..., 1] = vy
    vectors[..., 2] = vz
    # vectors = vectors.transpose(3, 0, 1, 2).copy()

    # arranges rgb data into correct format
    values = np.empty(z.shape + (3,), dtype=np.float32)
    values[..., 0] = val_major
    values[..., 1] = val_minor1
    values[..., 2] = val_minor2
    # values = values.transpose(3, 0, 1, 2).copy()

    print('Finished reshaping, now saving data...')

    results_memory_size = (sys.getsizeof(pts) / 1e6) * 3  # estimates memory size of results and converts to mb
    if results_memory_size >= 5000:  # if results data is larger than 5000mb (5gb), save results as multiple files
        multi_file = True
        split_amount = math.ceil(results_memory_size / 5000)  # number of sections/files to split the data into
        print('\033[93m' + f'Results too large for 1 file, saving results in {split_amount} files' + '\033[0m')
    else:
        multi_file = False

    if multi_file:
        # split data into subsections along the x-axis
        pts_split = np.array_split(pts, split_amount, axis=0)
        vectors_split = np.array_split(vectors, split_amount, axis=0)
        values_split = np.array_split(values, split_amount, axis=0)

        t1 = time.perf_counter()  # start time
        for i in range(split_amount):
            print(f'Saving file {i+1} of {split_amount}')
            pts = pts_split[i].transpose(2, 1, 0, 3).copy()
            pts.shape = pts.size // 3, 3

            vectors = vectors_split[i].transpose(2, 1, 0, 3).copy()
            vectors.shape = vectors.size // 3, 3

            values = values_split[i].transpose(2, 1, 0, 3).copy()
            values.shape = values.size // 3, 3

            sg = tvtk.StructuredGrid(dimensions=pts_split[i].shape[:3], points=pts)
            sg.point_data.vectors = vectors
            sg.point_data.vectors.name = 'Vectors'
            sg.point_data.add_array(values)
            sg.point_data.get_array(1).name = 'RGB'

            save_filename_split = save_filename + '_File' + str(i+1)  # generates unique file name
            write_data(sg, save_filename_split)
        t2 = time.perf_counter()  # stop time

    else:
        pts = pts.transpose(2, 1, 0, 3).copy()
        pts.shape = pts.size // 3, 3

        vectors = vectors.transpose(2, 1, 0, 3).copy()
        vectors.shape = vectors.size // 3, 3

        values = values.transpose(2, 1, 0, 3).copy()
        values.shape = values.size // 3, 3

        sg = tvtk.StructuredGrid(dimensions=x.shape, points=pts)

        sg.point_data.vectors = vectors
        sg.point_data.vectors.name = 'Vectors'
        sg.point_data.add_array(values)
        sg.point_data.get_array(1).name = 'RGB'

        t1 = time.perf_counter()  # start time
        write_data(sg, save_filename)
        t2 = time.perf_counter()  # stop time

    print(f'Data saved to file in {t2-t1} seconds')
    print('Finished - Program End')