"""
3D_Data_Processing
Version 6.0
Date: 31/08/2022
Created by Simon Burt in accordance with the MSc Individual Project
Key feature of this version - Ability to read nifti files for processing
"""
import time
import sys
import os
import math
import numpy as np
import tkinter
from tkinter import filedialog
import skimage.io
import skimage.transform
import multiprocessing as mp
from tqdm import tqdm
from structure_tensor import parallel_structure_tensor_analysis
from pyevtk.hl import gridToVTK
import matplotlib.pyplot as plt
import nibabel as nib
import tifffile


def write_tvk_file(coh, vec, val, step, save_filename, order, chunk_z_spacing, sigma, rho, pool):
    # function to shape results into the correct format for writing to a tvk file.
    # input:
    #       coh = 3-d numpy array containing the calculated coherence values.
    #       vec = 4-d numpy array containing calculated eigenvector results.
    #       val = 3-d numpy array containing calculated eigenvalue results.
    #       step = integer which determines the amount to downsample the data by. e.g. 1 would mean no down-sampling
    #       save_filename = string containing location where and name to save the results.
    #       order = integer indicating the position of the chunk of data that is being saved.
    #       chunk_z_spacing = 1-d tuple array containing the z-dimension of the every chunk.
    #       sigma = integer, single value used for calculating st
    #       rho = integer, single value used for calculating st

    nth_term = step     # how many terms to downsample data by

    # eigenvectors
    vx = vec[2, :, ::nth_term, ::nth_term]
    vy = vec[1, :, ::nth_term, ::nth_term]
    vz = vec[0, :, ::nth_term, ::nth_term]
    dim = vx.shape

    # rgb data
    val_major = val[..., 0]     # red channel
    val_minor1 = val[..., 1]    # green channel
    val_minor2 = val[..., 2]    # blue channel

    coh = coh[:, ::nth_term, ::nth_term]
    coherence = coh.transpose(0, 1, 2).copy()

    # generates grid and arranges into correct format.
    if order == 0:
        x, y, z = np.mgrid[0:dim[0], 0:dim[1], 0:dim[2]]
    else:
        start = 0
        for j in range(order):
            start = start + chunk_z_spacing[j]
        x, y, z = np.mgrid[start:(start+dim[0]), 0:dim[1], 0:dim[2]]
    pts = np.empty(z.shape + (3,), dtype=np.float32)
    pts[..., 0] = x
    pts[..., 1] = y
    pts[..., 2] = z
    pts = pts.transpose(3, 0, 1, 2).copy()  # re-orders matrix so x first, y next and z last.

    # arranges vector field data into correct format
    vectors = np.empty(z.shape + (3,), dtype=np.float32)
    vectors[..., 0] = vx
    vectors[..., 1] = vy
    vectors[..., 2] = vz
    vectors = vectors.transpose(3, 0, 1, 2).copy()

    # arranges eigen-value data into correct format
    values = np.empty(z.shape + (3,), dtype=np.float32)
    values[..., 0] = val_major
    values[..., 1] = val_minor1
    values[..., 2] = val_minor2
    values = values.transpose(3, 0, 1, 2).copy()

    # estimate size of results data to determine number of files to save processed data into
    results_memory_size = (sys.getsizeof(pts)/ 1e6)*3  # estimates memory size of results and converts to mb
    if results_memory_size >= 5000:       # if results data is larger than 5000mb (5gb), save results as multiple files
        multi_file = True
        split_amount = math.ceil(results_memory_size / 5000)  # number of sections/files to split the data into
        print('\033[93m' + f'results too large for 1 file, saving results in {split_amount} files' + '\033[0m')
    else:
        multi_file = False

    if multi_file:
        # save data as multiple tvk files if too large
        # split data into sub-sections along the x-axis
        pts_split = np.array_split(pts, split_amount, axis=1)
        vectors_split = np.array_split(vectors, split_amount, axis=1)
        values_split = np.array_split(values, split_amount, axis=1)
        coherence_split = np.array_split(coherence, split_amount, axis=0)

        # create list of tuples containing split data
        data= [None]*split_amount   # create empty list
        for ij in range(split_amount):
            save_filename_split = save_filename + "_processed_sigma=" + str(sigma) + "_rho=" + str(rho) + "_chunk" + str(order+1) +\
                                  "_file" + str(ij+1)  # generates unique file name
            itr = int(ij)
            data[ij] = (pts_split[ij], vectors_split[ij], values_split[ij], coherence_split[ij], save_filename_split, itr)

        for ik in tqdm(range(split_amount)):
            tvk_write(data[ik])

    else:
        save_filename = save_filename + "_processed_sigma=" + str(sigma) + "_rho=" + str(rho) + "_chunk" + str(order + 1)  # generates unique file name
        data = [None]
        data = (pts, vectors, values, coherence, save_filename, 0)
        tvk_write(data)


def tvk_write(data):
    # function which calls gridToVTK and writes data to file
    vectors_split_tuple = tuple(data[1])  # splits vector data into x,y and z matrices
    values_split_tuple = tuple(data[2])  # splits vector data into x,y and z matrices
    gridToVTK(data[4],      # filename
              data[0][0],   # x-data
              data[0][1],   # y-data
              data[0][2],   # z-data
              start=(vectors_split_tuple[0].shape[0] * data[5], 0, 0),
              pointdata={"vectors": vectors_split_tuple, "RGB": values_split_tuple, "coherence": data[3]}
              )

def nifti_write(volume, data, save_filename, sigma, rho, split_axis):
    # Function saves results in a new directory as a nifti files
    # Input:
    #       volume = 3-d numpy array containing greyscale image
    #       data = tuple consisting of multiple dicts
    #       save_filename = directory to save the data in
    #       sigma = int of the value used to process data
    #       rho = int of value used to process data

    nib.openers.Opener.default_compresslevel = 6    # Sets data compression level (range=1-9)
    p_bar = tqdm(range(7))      # Progress bar
    # Makes new directory
    path = save_filename
    os.mkdir(path)

    # Save volume data
    volume_img = nib.Nifti1Image(volume, affine=np.eye(4))
    volume_img.set_data_dtype(np.uint8)
    nib.save(volume_img, path + "/data_Vol_sigma=" + str(sigma) + "_rho=" + str(rho)+ ".nii.gz")
    p_bar.n = 1
    p_bar.refresh()
    del volume_img, volume

    # Save FA
    fa = np.concatenate([x['fa'] for x in data], axis=split_axis)
    # fa *= 255   # scale data to unit8
    fa_img = nib.Nifti1Image(fa, affine=np.eye(4))
    fa_img.set_data_dtype(np.uint8)
    nib.save(fa_img, path + '/data_FA.nii.gz')
    p_bar.n = 2
    p_bar.refresh()
    del fa_img, fa

    # Save V1 eigenvector
    vec = np.concatenate([x['vector'] for x in data], axis=split_axis+1)
    vec = vec.transpose(1, 2, 3, 0).copy()
    vec_img = nib.Nifti1Image(vec, affine=np.eye(4))
    vec_img.set_data_dtype(np.uint8)
    nib.save(vec_img, path + '/data_V1.nii.gz')
    p_bar.n = 3
    p_bar.refresh()
    del vec_img, vec

    # Save eigenvalues
    val = np.concatenate([x['values'] for x in data], axis=split_axis+1)
    val = val.transpose(1, 2, 3, 0).copy()
    tmp = np.subtract(val, np.min(val))     # Normalise 0 and 1
    tmp2 = np.max(val) - np.min(val)
    val = np.divide(tmp, tmp2)
    val *= 255  # scale data to uint8

    L1_img = nib.Nifti1Image(val[..., 0], affine=np.eye(4))
    L1_img.set_data_dtype(np.uint8)
    nib.save(L1_img, path + '/data_L1.nii.gz')
    del tmp, tmp2, L1_img
    p_bar.n = 4
    p_bar.refresh()

    L2_img = nib.Nifti1Image(val[..., 1], affine=np.eye(4))
    L2_img.set_data_dtype(np.uint8)
    nib.save(L2_img, path + '/data_L2.nii.gz')
    p_bar.n = 5
    p_bar.refresh()
    del L2_img

    L3_img = nib.Nifti1Image(val[..., 2], affine=np.eye(4))
    L3_img.set_data_dtype(np.uint8)
    nib.save(L3_img, path + '/data_L3.nii.gz')
    p_bar.n = 6
    p_bar.refresh()
    del L3_img, val

    # Save coherency
    coh = np.concatenate([x['coh'] for x in data], axis=split_axis)
    tmp = np.subtract(coh, np.min(coh)) # norlaise data between 0 and 1
    tmp2 = np.max(coh)-np.min(coh)
    coh = np.divide(tmp, tmp2)
    coh *= 255   # scale data to uint8

    coh_img = nib.Nifti1Image(coh, affine=np.eye(4))
    coh_img.set_data_dtype(np.uint8)
    nib.save(coh_img, path + '/data_COH.nii.gz')
    del tmp, tmp2, coh_img, coh
    p_bar.n = 7; p_bar.refresh()

def coherence(val):
    # function calculates the coherence value of every voxel
    # Based on the following formula:
    # c_s = (3 * eigval3) / (eigval1 ** 2 + eigval2 ** 2 + eigval3 ** 2) ** 0.5

    # Flatten S.
    input_shape = val.shape
    val = val.reshape(3, -1)
    c_s = np.empty((1,) + val.shape[1:], dtype="float32")
    c_a = np.empty((1,) + val.shape[1:], dtype="float32")
    tmp = np.empty((4,) + val.shape[1:], dtype="float32")

    # compute c_s
    np.multiply(val[2], 3, out=tmp[0])         # 3 * eigval3
    np.multiply(val[0], val[0], out=tmp[1])   # eigval1^2
    np.multiply(val[1], val[1], out=tmp[2])   # eigval2^2
    np.multiply(val[2], val[2], out=tmp[3])   # eigval3^2
    a = np.add(tmp[1], tmp[2])
    a += tmp[3]
    a **= 0.5
    np.divide(tmp[0], a, out=c_s, where=a != 0)
    # compute c_a
    np.subtract(1, c_s, out=c_a)
    del tmp
    return c_a.reshape(input_shape[1:])

def fractional_anisotropy(val):
    # function calculates fractional anisotropy
    # Based on the following formula:
    # fa = (1/2)**0.5 * ((eigval1 - eigval2)**2 + (eigval2-eigval3)**2 + (eigval3 - eigval1)**2)** 0.5/(eigval1 ** 2 + eigval2 ** 2 + eigval3 ** 2)) ** 0.5

    # Flatten val
    input_shape = val.shape
    val = val.reshape(3, -1)
    fa = np.empty((1,) + val.shape[1:], dtype="float32")
    tmp = np.empty((5,) + val.shape[1:], dtype="float32")

    # compute fa
    np.multiply(val[0], val[0], out=tmp[0])   # eigval1^2
    np.multiply(val[1], val[1], out=tmp[1])   # eigval2^2
    np.multiply(val[2], val[2], out=tmp[2])   # eigval3^2
    np.add(tmp[0], tmp[1], out=tmp[3])
    tmp[3] += tmp[2]

    a = np.subtract(val[0], val[1])
    a **= 2
    b = np.subtract(val[1], val[2])
    b **= 2
    c = np.subtract(val[2], val[0])
    c **= 2
    np.add(a, b, out=tmp[4])
    tmp[4] += c

    d = np.divide(tmp[4], tmp[3], where=tmp[3] != 0)
    d **= 0.5
    e = 1/(2**0.5)

    np.multiply(e, d, out=fa)
    del tmp
    return fa.reshape(input_shape[1:])


def colouring(vec, fa):
    # function to assign rgba value to data
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
    b = np.divide(tmp[2], tmp[3])  # x-axis component is the red channel
    # find green
    np.subtract(vec[0], np.nanmin(vec[0]), out=tmp[4])
    np.subtract(np.nanmax(vec[0]), np.nanmin(vec[0]), out=tmp[5])
    g = np.divide(tmp[4], tmp[5])  # x-axis component is the red channel
    a = fa  # use fa as alpha channel

    rgba = np.empty((input_shape[1], input_shape[2], input_shape[3], 4))
    rgba[..., 0] = r.reshape(input_shape[1:])
    rgba[..., 1] = g.reshape(input_shape[1:])
    rgba[..., 2] = b.reshape(input_shape[1:])
    rgba[..., 3] = a
    # rgba *= 255  # normalise data
    rgba = rgba.astype(np.uint8)  # convert to uint8 to reduce file size
    del tmp
    return rgba

def array_slice(a, axis, start, end, step=1):
    return a[(slice(None),) * (axis % a.ndim) + (slice(start, end, step),)]

def chunk_split(volume, split_amount, padding):
    # function to split large datasets into smaller manageable chunks for processing
    # input:
    #       volume = numpy array of the dataset to be split into chunks.
    #       split_amount = integer of the number of chunks to split the data into.
    #       padding = integer of the number of layers to add to each chunk. takes the data from the neighbouring chunk.
    #                 if first or last chunk, padding will only be added to one side of the chunk.
    # output:
    #       volume_chunk = list containing multiple numpy arrays of the chunks.
    #       chunk_amount_z = tuple containing the z-dimension of the every chunk the data was split into.

    # Determine largest axis and split data along it
    if volume.shape[0] >= volume.shape[1] and volume.shape[0] >= volume.shape[2]:
        split_axis = 0
    elif volume.shape[1] >= volume.shape[0] and volume.shape[1] >= volume.shape[2]:
        split_axis = 1
    else:
        split_axis = 2

    chunk_amount_z = tuple([volume.shape[split_axis] // split_amount + int(x < volume.shape[split_axis] % split_amount) for x in range(split_amount)])
    volume_chunks = [None] * split_amount  # initialise variable
    for iz in range(split_amount):
        if iz == 0:
            # deals with initial chunk
            volume_chunks[iz] = {
                                'data': array_slice(volume, split_axis, 0, (chunk_amount_z[iz] + padding)),
                                'order': int(iz),
                                'padding_start': 0,
                                'padding_end': padding
                                }
            previous_finish = chunk_amount_z[iz]
        elif iz == (split_amount - 1):
            # deals with last chunk
            volume_chunks[iz] = {
                                'data': array_slice(volume, split_axis, (previous_finish - padding), (previous_finish + chunk_amount_z[iz])),
                                'order': int(iz),
                                'padding_start': padding,
                                'padding_end': 0
                                }
        else:
            # deals with all other chunks
            volume_chunks[iz] = {
                                'data': array_slice(volume, split_axis, (previous_finish - padding),(previous_finish + chunk_amount_z[iz] + padding)),
                                'order': int(iz),
                                'padding_start': padding,
                                'padding_end': padding
                                }
            previous_finish = previous_finish + chunk_amount_z[iz]
    return(volume_chunks, chunk_amount_z, split_axis)


if __name__ == '__main__':

    sigma = 1 # noise scale
    rho = 10 # integration scale
    save_as_tvk = False    # True - if want to save results as tvk and nifti file (program takes longer to run)

    # hides root window
    root = tkinter.Tk()
    root.withdraw()

    # ask user to select image and load image using ui
    filename = filedialog.askopenfile(title='select file to process')
    if filename.name[-4:]=='.nii' or filename.name[-4:]=='i.gz':
        # Reads nifti files
        data_info = nib.load(filename.name)
        volume = np.asarray(data_info.dataobj, dtype='float32')
    else:
        # Reads tiff files
        print('import data...')
        volume = skimage.io.imread(filename.name)   # volume shape = (z,y,x)
        volume = volume.transpose(2, 1, 0)   # reshape to (x,y,z)
        if volume.dtype == 'uint16':
            # convert image to 8-bit greyscale image if it is 16-bit
            print('Data imported is of type uint16, converting to uint8')
            volume /= 256
            volume = volume.astype('unit8')
    print(f'data size = {volume.shape}')

    # check sigma and rho is no more than 10% of data size
    smallest_size = min(volume.shape[0], volume.shape[1], volume.shape[2])
    # check sigma value is smaller than 10% of image size
    if not smallest_size * 0.1 >= sigma:
        sys.exit('sigma value must be smaller than 10% of volume size in all planes')
    # check rho value is smaller than 10% of image size
    if not smallest_size * 0.1 >= rho:
        sys.exit('rho value must be smaller than 10% of volume size in all planes')

    # plot 2d slice of data in 3-planes
    show_slice= False
    if show_slice:
        plt.figure()
        fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        skimage.io.imshow(volume[0, :, :], ax=ax[0])    # x-axis
        skimage.io.imshow(volume[:, 0, :], ax=ax[1])    # y-axis
        skimage.io.imshow(volume[:, :, 0], ax=ax[2])    # z-axis

    # ask user location to save results
    save_filename = filedialog.asksaveasfilename(title='select location and enter name of file to save results')
    root.destroy()
    if save_filename == '':  # asksaveasfile return `none` if dialog closed with "cancel".
        sys.exit('no file name or location was selected to save results')

    # check to see if data needs splitting into chunks for processing
    if volume.size >= 160000000:
        # split data into smaller chunks for processing + add padding to avoid boundary errors
        chunked_data = True
        split_amount = math.ceil(volume.size / 160000000)  # number of chunks to split the data into
        print('\033[93m'+ f'Warning: Large dataset loaded. Data will be split into {split_amount} chunks for processing.' + '\033[0m')
        volume_chunks, chunk_amount_z, split_axis = chunk_split(volume, split_amount, rho)    # calls function to split data into manageable chunks
    else:
        chunked_data = False
        split_amount = 1
        order = 0
        chunk_amount_z = volume.shape[2]
        split_axis = 0

    # start pool for multicore processing
    if mp.cpu_count()>=60:
        pool = mp.Pool(mp.cpu_count()-10)   # error occurs if pc has more than 60-cores
    else:
        pool = mp.Pool(mp.cpu_count())

    # process data
    final_data = [None] * split_amount  # create empty list
    for inds in range(split_amount):
        if chunked_data:
            chunk = volume_chunks[inds]['data']
            order = volume_chunks[inds]['order']
            padding_start = volume_chunks[inds]['padding_start']
            padding_end = volume_chunks[inds]['padding_end']
        else:
            chunk = volume.astype('float')

        # calculate structure tensor,eigenvalues and eigenvectors
        print(f'calculating structure tensor for chunk {order+1}...')
        t1 = time.perf_counter()  # start time
        s, vec, val = parallel_structure_tensor_analysis(chunk, sigma, rho,
                                                         structure_tensor=True) # vec has shape =(3,x,y,z) in the order of (z,y,x)

        # remove padding from st, e'values & e'vectors if data is chunked
        if chunked_data:
            array_end = s.shape[split_axis+1] - padding_end
            s = array_slice(s, split_axis+1, padding_start, array_end)
            vec = array_slice(vec, split_axis+1, padding_start, array_end)
            val = array_slice(val, split_axis+1, padding_start, array_end)

        t2 = time.perf_counter()  # stop time
        del s   # save some memory as no longer needed
        print(f'finished calculating structure tensors in {t2 - t1} seconds')

        # call function to calculate coherency
        print(f'calculating coherence values for chunk {order+1} ...')
        t1 = time.perf_counter()  # start time
        val_chunks = np.array_split(val, val.shape[3], axis=3)      # split array into chunks to be parallel processed
        coh = list(tqdm(pool.imap(coherence, [chunk_coh for chunk_coh in val_chunks]),
                        total=len(val_chunks)))      # calculates coherency from eigenvalues using parallel processing
        coh_result = np.concatenate(coh, axis=2)    # joins array chunks back into one
        del coh     # saves some memory
        t2 = time.perf_counter()        # stop time
        print(f'finished calculating coherence in {t2 - t1} seconds')

        # call function to calculate fractional anisotropy (fa)
        print(f'calculating fa values for chunk {order + 1} ...')
        t1 = time.perf_counter()  # start time
        fa = list(tqdm(pool.imap(fractional_anisotropy, [chunk_coh for chunk_coh in val_chunks]),
                        total=len(val_chunks)))  # calculates coherency from eigenvalues using parallel processing
        fa_result = np.concatenate(fa, axis=2)  # joins array chunks back into one
        del fa      # saves some memory
        t2 = time.perf_counter()  # stop time
        print(f'finished calculating fa in {t2 - t1} seconds')

        # calls function to assign rgb value to each eigenvector and display results
        # show_vol_orientation(chunk, vec, fa_result, coloring=colouring)
        if save_as_tvk:
            rgba = colouring(vec, fa_result)
            rgb = np.zeros((rgba.shape[0], rgba.shape[1], rgba.shape[2], 3))
            rgb[..., 0] = rgba[..., 0]       # red channel
            rgb[..., 1] = rgba[..., 1]       # green channel
            rgb[..., 2] = rgba[..., 2]       # blue channel
            rgb *= 255                      # normalise data
            rgb = rgb.astype(np.uint8)  # convert to uint8 to reduce file size
        else:
            rgb = np.nan

        # store all data in a list
        final_data[inds] = {'coh': coh_result,
                            'vector': vec,
                            'values': val,
                            'order': order,
                            'chunk_z_spacing': chunk_amount_z,
                            'fa': fa_result,
                            'rgb': rgb}
        del coh_result, vec, val, fa_result, rgb    # save some memory by removing duplicate data

    # write results to file
    print('saving results to file...')
    if save_as_tvk:
        # saves data as tvk file(s)
        for ijj in tqdm(range(split_amount)):
           write_tvk_file(final_data[ijj]['coh'], final_data[ijj]['vector'], final_data[ijj]['rgb'], 1, save_filename,
                           final_data[ijj]['order'], final_data[ijj]['chunk_z_spacing'], sigma, rho, pool)

    nifti_write(volume, final_data, save_filename, sigma, rho, split_axis)    # Saves data in nifti format



    print(f'data saved in: {save_filename}')
    pool.close()
    pool.join()
    print('finished - program end')
    plt.show()