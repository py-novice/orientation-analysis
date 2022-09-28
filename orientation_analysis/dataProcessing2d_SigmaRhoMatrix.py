"""Function to plot a matrix of different sigma and rho values
Version 2 - Allows an array of no consecutive values for sigma & rho
"""

import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tkinter
from tkinter import filedialog
from scipy import stats
from structure_tensor import eig_special_2d, structure_tensor_2d
from plot_orientations import plot_orientations
from rgb2grey import rgb2grey
from tqdm import tqdm
import multiprocessing as mp

def coherence(S):
    # function calculates the coherence value of every pixel

    epsilon = 0.001
    # Flatten S.
    input_shape = S.shape
    s = S.reshape(3,-1)
    tmp = np.empty((4,) + s.shape[1:], dtype="float32")

    np.subtract(s[1], s[0], out=tmp[0])
    tmp[0] **= 2
    np.multiply(s[2], 4, out=tmp[1])
    tmp[1] **= 2
    np.add(tmp[1], tmp[0], out=tmp[2])
    np.add(s[1], s[0], out=tmp[3])
    tmp[3] += epsilon
    coherence = np.divide(tmp[2], tmp[3])

    return coherence.reshape(input_shape[1:])

def orientation(S):
    # function to calculate vector orientation
    # Flatten S.
    input_shape = S.shape
    s = S.reshape(3, -1)
    tmp = np.empty((4,) + s.shape[1:], dtype="float32")

    np.multiply(s[2], 2, out=tmp[0])
    np.subtract(s[1], s[0], out=tmp[1])
    tmp[2] = np.arctan2(tmp[0], tmp[1])
    np.multiply(tmp[2], 0.5, out=tmp[3])
    ori = np.degrees(tmp[3])

    return ori.reshape(input_shape[1:])

def energy(val):
    # function to calculate energy

    input_shape = val.shape
    val = val.reshape(2, -1)
    eng = np.add(val[0], val[1])
    return eng.reshape(input_shape[1:])

if __name__ == '__main__':
    root = tkinter.Tk()
    root.wm_withdraw()  # hides root window

    # Initialise variables
    SigmaValues = [1,2,3,4,5,6,7,8,9,10]    # Enter vector of sigma and rho values
    epsilon = 0.001     # coherency
    vector_spacing = 40     # vector field spacing when plotted
    neighborhood_radius = 10    # Window size for calculating order parameter
    iters = len(SigmaValues)

    # Load image using ui
    filename = filedialog.askopenfile()
    image = mpimg.imread(filename.name)

    rows = image.shape[0]       # gets image size
    columns = image.shape[1]
    if len(image.shape)>=3:     # checks if image is greyscale
        image = rgb2grey(image)

    # Check sigma and rho is no more than 10% of image size
    if rows >= columns:
        smallest_size = columns
    else:
        smallest_size = rows

    for ix in SigmaValues:
        if smallest_size*0.1 >= ix:   # checks all sigma values are smaller than 10% of image size
            print('Sigma value good')
        else:
            sys.exit('Sigma Values must be smaller than 10% of the image size')


    # Calculate structure tensor & eigenvalues for different sigma and rho values
    figsize = (9, 7)      # Sets size of windows (width, length)
    fig1, ax = plt.subplots(iters, iters, figsize=figsize, sharex=True, sharey=True)
    fig2, ax2 = plt.subplots(iters, iters, figsize=figsize, sharex=True, sharey=True)
    fig3, ax3 = plt.subplots(iters, iters, figsize=figsize, sharex=True, sharey=True)
    fig4, ax4 = plt.subplots(iters, iters, figsize=figsize, sharex=True, sharey=True)


    median_OrderPara = np.zeros((iters, iters))
    trim_mean = np.zeros((iters, iters))
    print('Calculating structure tensors...')
    p_bar = tqdm(range(iters*iters))      # Progress bar
    counter = 0
    for i in range(0, iters):
        for j in range(0, iters):

            sigma = SigmaValues[i]
            rho = SigmaValues[j]
            S = structure_tensor_2d(image.astype('float'), sigma, rho)
            val, vec = eig_special_2d(S)


            # Calculate coherence
            coh_result = coherence(S)

            # Calculate orientation
            ori_result = orientation(S)

            # Calculate energy
            eng_result = energy(val)

            ax[i,j].imshow(image, cmap=plt.cm.gray)
            plot_orientations(ax[i, j], image.shape, vec, vector_spacing)
            ax[i,j].set_axis_off()

            ax2[i, j].imshow(coh_result)
            ax2[i,j].set_axis_off()

            flat_coh = coh_result.reshape(np.prod(coh_result.shape[:2], -1))
            flat_ori = ori_result.reshape(np.prod(ori_result.shape[:2], -1))
            x_bins = range(0,20,1)
            y_bins = np.arange(-1,1,0.1)
            ax3[i, j].hist(flat_ori, bins=range(-90,90,1))     # Plots histogram of orientation values
            # ax3[i, j].hist(flat_coh, bins=180, density=True)         # Plots histogram of coherency values
            # ax3[i, j].hist2d(flat_coh, flat_order, bins=[x_bins, y_bins])
            ax3[i, j].get_yaxis().set_visible(False)
            ax3[i, j].set_xlim((-90, 90))

            ax4[i, j].imshow(eng_result)
            ax4[i, j].set_axis_off()

            # Calculates trimmed mean
            ori_bins = np.arange(-90, 90, 1)  # Bins sizes to arrange orientation values into
            freq, bin_edges = np.histogram(ori_result,
                                           ori_bins)  # Calculates the number of times the orientation appears in each bin
            trim_mean[i,j] = stats.trim_mean(freq, 0.05) # Orders freq from smallest to largest and removes the
                                                        # top and bottom 5% (outliers), then calculates the mean
            trim_mean[i,j] = trim_mean[i,j]/((rows*columns)/180)    # Nomralise mean between 0-1

            counter = counter+1
            p_bar.n = counter
            p_bar.refresh()
            # Calculates order parameter
    """
            # Code extract taken from
            # https://github.com/OakesLab/AFT-Alignment_by_Fourier_Transform/blob/master/Python_implementation/AFT_tools.py
            
            order_list = []
            neighborhood_radius = 10
            rpos = np.arange(neighborhood_radius, orientation.shape[0] - neighborhood_radius)
            cpos = np.arange(neighborhood_radius, orientation.shape[1] - neighborhood_radius)
    
            for r in rpos:
                for c in cpos:
                    search_window = orientation[r - neighborhood_radius:r + neighborhood_radius + 1,
                                    c - neighborhood_radius:c + neighborhood_radius + 1]
                    vector_array = np.ones_like(search_window) * orientation[r, c]
                    order_array = np.cos(search_window - vector_array) ** 2 - 0.5
                    order_array = np.delete(order_array, (2 * neighborhood_radius + 1) ** 2 // 2)
                    if not np.isnan(order_array).all() == True:
                        if order_array.size > 0:
                            order_list.append(2 * np.nanmean(order_array))
    
            ax3[i, j].hist(order_list, bins=np.arange(-1, 1, 0.01))  # Plots histogram of order parameter values
            ax3[i, j].set_xlim((-1, 1))
            median_OrderPara[i, j] = np.median(order_list)
    """
    # Common axis labels - Vector Field
    fig = ax[0][0].get_figure()  # getting the figure
    ax = fig.add_subplot(111, frame_on=False)   # creating a single axes
    # x-axis labels and tick marks
    ax.set_xticks(np.arange(1,iters+1,1))
    ax.set_xticks([float(n)-0.5 for n in ax.get_xticks()])
    ax.set_xticklabels(SigmaValues, fontsize=12)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Rho', fontsize=15, labelpad=10)
    # y-axis labels and tick marks
    ax.set_yticks(np.arange(1,iters+1,1))
    ax.set_yticks([float(n)-0.5 for n in ax.get_yticks()])
    ax.set_yticklabels(SigmaValues, fontsize=12)
    ax.invert_yaxis()
    ax.set_ylabel('Sigma', fontsize=15, labelpad=10)

    # Common axis labels - Coherency
    fig = ax2[0][0].get_figure()  # getting the figure
    ax2 = fig.add_subplot(111, frame_on=False)   # creating a single axes
    # x-axis labels and tick marks
    ax2.set_xticks(np.arange(1,iters+1,1))
    ax2.set_xticks([float(n)-0.5 for n in ax2.get_xticks()])
    ax2.set_xticklabels(SigmaValues, fontsize=12)
    ax2.xaxis.set_ticks_position('top')
    ax2.xaxis.set_label_position('top')
    ax2.set_xlabel('Rho', fontsize=15, labelpad=10)
    # y-axis labels and tick marks
    ax2.set_yticks(np.arange(1,iters+1,1))
    ax2.set_yticks([float(n)-0.5 for n in ax2.get_yticks()])
    ax2.set_yticklabels(SigmaValues, fontsize=12)
    ax2.invert_yaxis()
    ax2.set_ylabel('Sigma', fontsize=15, labelpad=10)


    # Common axis labels - Coherency histogram
    fig = ax3[0][0].get_figure()  # getting the figure
    ax3 = fig.add_subplot(111, frame_on=False)   # creating a single axes
    # x-axis labels and tick marks
    ax3.set_xticks(np.arange(1,iters+1,1))
    ax3.set_xticks([float(n)-0.5 for n in ax3.get_xticks()])
    ax3.set_xticklabels(SigmaValues, fontsize=12)
    ax3.xaxis.set_ticks_position('top')
    ax3.xaxis.set_label_position('top')
    ax3.set_xlabel('Rho', fontsize=15, labelpad=10)
    # y-axis labels and tick marks
    ax3.set_yticks(np.arange(1,iters+1,1))
    ax3.set_yticks([float(n)-0.5 for n in ax3.get_yticks()])
    ax3.set_yticklabels(SigmaValues, fontsize=12)
    ax3.invert_yaxis()
    ax3.set_ylabel('Sigma', fontsize=15, labelpad=10)

    # Common axis labels - energy
    fig = ax4[0][0].get_figure()  # getting the figure
    ax4 = fig.add_subplot(111, frame_on=False)   # creating a single axes
    # x-axis labels and tick marks
    ax4.set_xticks(np.arange(1,iters+1,1))
    ax4.set_xticks([float(n)-0.5 for n in ax4.get_xticks()])
    ax4.set_xticklabels(SigmaValues, fontsize=12)
    ax4.xaxis.set_ticks_position('top')
    ax4.xaxis.set_label_position('top')
    ax4.set_xlabel('Rho', fontsize=15, labelpad=10)
    # y-axis labels and tick marks
    ax4.set_yticks(np.arange(1,iters+1,1))
    ax4.set_yticks([float(n)-0.5 for n in ax4.get_yticks()])
    ax4.set_yticklabels(SigmaValues, fontsize=12)
    ax4.invert_yaxis()
    ax4.set_ylabel('Sigma', fontsize=15, labelpad=10)

    # Reduce white space between images and label figures
    fig1.tight_layout(); fig1.canvas.set_window_title('Vector Field')
    fig2.tight_layout(); fig2.canvas.set_window_title('Coherency')
    fig3.tight_layout(); fig3.canvas.set_window_title('Coherency Histogram')
    fig4.tight_layout(); fig4.canvas.set_window_title('Energy')
    print('FINISHED')
    print(trim_mean)

    # Plot trimmed mean
    fig5, ax5 = plt.subplots(1,1)
    trim_mean_plot =ax5.imshow(trim_mean, vmin=0, vmax=1) #uncomment to plot colour bar between 0 and 1
    # trim_mean_plot =ax5.imshow(trim_mean)
    # x-axis labels and tick marks
    ax5.set_xticks(np.arange(1,iters,1))
    ax5.set_xticks([float(n)-0.5 for n in ax3.get_xticks()])
    ax5.set_xticklabels(SigmaValues, fontsize=11)
    ax5.xaxis.set_ticks_position('top')
    ax5.xaxis.set_label_position('top')
    ax5.set_xlabel('Rho', fontsize=15, labelpad=10)
    # y-axis labels and tick marks
    ax5.set_yticks(np.arange(1,iters,1))
    ax5.set_yticks([float(n)-0.5 for n in ax3.get_yticks()])
    ax5.set_yticklabels(SigmaValues, fontsize=11)
    ax5.set_ylabel('Sigma', fontsize=15, labelpad=10)
    # Add colour bar
    fig5.colorbar(trim_mean_plot)
    # Add figure title
    fig5.canvas.set_window_title('Trimmed Mean')
    # Find smallest trimmed mean and print parameters used to achieve it
    min_idx = np.where(trim_mean==np.amin(trim_mean))
    print('Optimum results achieved when:')
    print('Sigma = ',SigmaValues[min_idx[0][0]])
    print('Rho = ',SigmaValues[min_idx[1][0]])

    plt.show()
    root.mainloop()