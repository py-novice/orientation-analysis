"""
niftiToRGB
Version 2.0
Date: 31/08/2022
Created by Simon Burt in ful
"""
import sys
import time
import numpy as np
import nibabel as nib
import tkinter
from tkinter import filedialog
from tkinter.messagebox import askyesno
import tifffile
from tqdm import tqdm
import multiprocessing as mp

def convert2rgb(vec):
    tmp = np.empty((2,) + vec.shape[0:], dtype="float16")
    vec = abs(vec)  # negative numbers don't matter for assigning colour
    vec[~np.isfinite(vec)] = np.nan     # replace and inf values with nan
    np.subtract(vec, np.nanmin(vec), out=tmp[0])
    np.subtract(np.nanmax(vec), np.nanmin(vec), out=tmp[1])
    tmp[0] /= tmp[1]
    return(tmp[0])

if __name__ == '__main__':
    # hides root window
    root = tkinter.Tk()
    root.withdraw()

    # ask user to select image and load image using ui
    filename = filedialog.askopenfile(title='select vector file to process')
    answer = askyesno(title='Include FA Data?', message='Do you want to include FA data?')
    if answer:
        filename_fa = filedialog.askopenfile(title='select FA file to process')


    # ask user location to save results
    save_filename = filedialog.asksaveasfilename(title='select location and enter name of file to save results')
    root.destroy()
    if save_filename == '':  # asksaveasfile return `none` if dialog closed with "cancel".
        sys.exit('no file name or location was selected to save results')

    # Import nifti file
    print('Importing data...')
    data_info = nib.load(filename.name)
    # data = data_info.get_fdata()
    data = np.asarray(data_info.dataobj, dtype='float16')
    if answer:
        fa_data_info = nib.load(filename_fa.name)
        # fa = fa_data_info.get_fdata()
        fa = np.asarray(fa_data_info.dataobj, dtype='float16')
    print('Finished importing, now calculating RGB values...')
    t1 = time.perf_counter()  # start time
    eigVec = data.transpose(3, 0, 1, 2)  # re-orders data so it's (3,x,y,z)
    t2 = time.perf_counter()  # stop time
    print(f'finished transposing array in {t2 - t1} seconds')

    # Assign rgb values to vector data
    input_shape = eigVec.shape
    t1 = time.perf_counter()  # start time
    vec = eigVec.reshape(3, -1)
    t2 = time.perf_counter()  # stop time
    print(f'finished reshaping array in {t2 - t1} seconds')

    # start pool for multicore processing
    if mp.cpu_count()>=60:
        pool = mp.Pool(mp.cpu_count()-10)   # error occurs if pc has more than 60-cores
    else:
        pool = mp.Pool(mp.cpu_count())

    # find r
    print('Calculating Red values...')
    t1 = time.perf_counter()  # start time
    vec_chunks = np.array_split(vec[2], pool._processes, axis=0)  # split array into chunks to be parallel processed
    r = list(tqdm(pool.imap(convert2rgb, [chunk_coh for chunk_coh in vec_chunks]),
                    total=len(vec_chunks)))  # calculates red channel using parallel processing
    r_result = np.concatenate(r, axis=0)  # joins array chunks back into one
    del r, vec_chunks  # saves some memory
    t2 = time.perf_counter()  # stop time
    print(f'finished calculating red values in {t2 - t1} seconds')

    # find b
    print('Calculating Blue values...')
    t1 = time.perf_counter()  # start time
    vec_chunks = np.array_split(vec[0], pool._processes, axis=0)  # split array into chunks to be parallel processed
    b = list(tqdm(pool.imap(convert2rgb, [chunk_coh for chunk_coh in vec_chunks]),
                    total=len(vec_chunks)))  # calculates red channel using parallel processing
    b_result = np.concatenate(b, axis=0)  # joins array chunks back into one
    del b, vec_chunks  # saves some memory
    t2 = time.perf_counter()  # stop time
    print(f'finished calculating blue values in {t2 - t1} seconds')

    # find g
    print('Calculating Green values...')
    t1 = time.perf_counter()  # start time
    vec_chunks = np.array_split(vec[1], pool._processes, axis=0)  # split array into chunks to be parallel processed
    g = list(tqdm(pool.imap(convert2rgb, [chunk_coh for chunk_coh in vec_chunks]),
                    total=len(vec_chunks)))  # calculates red channel using parallel processing
    g_result = np.concatenate(g, axis=0)  # joins array chunks back into one
    del g, vec_chunks  # saves some memory
    t2 = time.perf_counter()  # stop time
    print(f'finished calculating green values in {t2 - t1} seconds')

    rgba = np.empty((input_shape[1], input_shape[2], input_shape[3], 3), dtype='float16')
    rgba[..., 0] = r_result.reshape(input_shape[1:])
    rgba[..., 1] = g_result.reshape(input_shape[1:])
    rgba[..., 2] = b_result.reshape(input_shape[1:])
    del r_result, g_result, b_result     # Once assigned delete to free up some space
    rgba = rgba.transpose(2, 0, 1, 3)
    print('Shaping RGB matrix...')
    rgba *= 255     # normalise data
    RGB = rgba.astype(np.uint8)  # convert to uint8 to reduce file size
    RGB = np.flipud(RGB)    # Orientate data so matches tiff stack
    RGB = np.fliplr(RGB)
    RGB = np.rot90(RGB, k=3, axes=(1,2))

    print('Saving images...')
    # save results as a tiff stack
    with tifffile.TiffWriter(save_filename +'.tif', bigtiff=True) as tif:
        for i in tqdm(range(RGB.shape[0])):
            filename_tiff = f"image_{i}"
            img = RGB[i,...]
            # tif.write(img, photometric='rgb', description=filename_tiff, metadata=None)
            tif.write(img, photometric='rgb', contiguous=True)


    print('Finished - End Program')