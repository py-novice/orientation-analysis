import sys
import time
import numpy as np
import nibabel as nib
import tkinter
from tkinter import filedialog
import utils_3D
import matplotlib.pyplot as plt
from tkinter.messagebox import askyesno

def cart2sph(x, y, z):

    tmp = np.empty((1,) + x.shape[0:], dtype="float16")
    azimuth = np.arctan2(y, x)
    x **= 2
    y **= 2
    np.add(x, y, out=tmp[0])
    # z **= 2
    # r = np.sqrt(tmp + z)
    tmp **= 0.5
    elevation = np.arctan2(z, tmp)

    return azimuth, elevation

if __name__ == '__main__':
    # hides root window
    root = tkinter.Tk()
    root.withdraw()

    # ask user to select vector data and load  using ui
    filename = filedialog.askopenfile(title='select vector file to process')
    answer = askyesno(title='Include 2nd data set?', message='Do you want to include a second data set?')
    if answer:
        filename2 = filedialog.askopenfile(title='select 2nd vector file to process')
    root.destroy()

    # Import nifti file
    print('Importing data...')
    data_info = nib.load(filename.name)
    data = np.asarray(data_info.dataobj, dtype='float16')

    print('Finished importing, now calculating orientations...')
    eigVec = data.transpose(3, 0, 1, 2)  # re-orders data so it's (3,x,y,z)

    t1 = time.perf_counter()  # start time
    vec = eigVec.reshape(3, -1)
    t2 = time.perf_counter()  # stop time
    print(f'Reshaped vector in {t2 - t1} seconds')

    # Convert vectors to spherical coordinates
    sphDir = np.empty([2, vec.shape[1]], dtype='float')
    sphDir[0, :], sphDir[1, :] = cart2sph(vec[0, :], vec[1, :], vec[2, :])

    # Plot azimuth angles
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    bin = np.linspace(-np.pi, np.pi, 360)
    ax1.hist(sphDir[0, :], bins=bin, histtype=u'step', density=True)
    ax1.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax1.set_xticklabels(['-pi', '-pi/2', '0', 'pi/2', 'pi'])
    ax1.set_title('Distribution of Azimuth Angles')
    ax1.set_xlabel('Azimuth (radians)')
    ax1.set_ylabel('Normalised Frequency')

    # Plot elevation angles
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
    bin2 = np.linspace(-np.pi/2, np.pi/2, 180)
    ax2.hist(sphDir[1, :], bins=bin2, histtype=u'step', density=True)
    ax2.set_xticks([-np.pi / 2, 0, np.pi / 2])
    ax2.set_xticklabels(['-pi/2', '0', 'pi/2'])
    ax2.set_title('Distribution of Elevation Angles')
    ax2.set_xlabel('Elevation (radians)')
    ax2.set_ylabel('Normalised Frequency')

    # Repeat all steps for the second set of data
    if answer:
        print('Importing 2nd data set...')
        data2_info = nib.load(filename2.name)
        data2 = np.asarray(data2_info.dataobj, dtype='float16')
        eigVec2 = data2.transpose(3, 0, 1, 2)  # re-orders data so it's (3,x,y,z)
        print('Reshaping 2nd data set...')
        vec2 = eigVec2.reshape(3, -1)
        print('Converting 2nd data set to spherical coordinates...')
        sphDir2 = np.empty([2, vec2.shape[1]], dtype='float')
        sphDir2[0, :], sphDir2[1, :] = cart2sph(vec2[0, :], vec2[1, :], vec2[2, :])
        print('Plotting 2nd data set...')
        ax1.hist(sphDir2[0, :], bins=bin, histtype=u'step', density=True)
        ax2.hist(sphDir2[1, :], bins=bin2, histtype=u'step', density=True)


    nbin = (128, 128)
    binval, binc_az, binc_ele = utils_3D.histogramSphere(vec, nbin)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(binval.T, cmap='hsv', extent=[-np.pi, np.pi, -np.pi / 2, np.pi / 2])
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels(['-pi', '-pi/2', '0', 'pi/2', 'pi'])
    ax.set_yticks([-np.pi / 2, 0, np.pi / 2])
    ax.set_yticklabels(['-pi/2', '0', 'pi/2'])
    ax.set_title('spherical histogram over angles')
    ax.set_xlabel('azimuth')
    ax.set_ylabel('inclination')
    # fig.colorbar(binc_az)

    print('Finished - Program End')
    plt.show()