"""2D Structor Tensors

Useful links: https://github.com/Skielex/structure-tensor
              https://lab.compute.dtu.dk/QIM/tutorials/structuretensor
"""

import wx
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from structure_tensor import eig_special_2d, structure_tensor_2d
from plot_orientations import plot_orientations
from rgb2grey import rgb2grey

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx as NavigationToolbar
from matplotlib.figure import Figure
from collections import defaultdict


class MyFrame(wx.Frame):
    sigma = []
    rho = []
    vector = []
    neighbourhood = []
    filepath = {}
    filenmae = {}

    def __init__(self):
        wx.Panel.__init__(self, parent=None,title='2D Structor Tensor', size=(200, 350))
        self.SetBackgroundColour("White")
        box = wx.BoxSizer(wx.VERTICAL)
        box.Add((15, 15))

        # Sigma slider
        self.sigma_sld = wx.Slider(self, value=10, minValue=1, maxValue=50,
                                   style=wx.SL_HORIZONTAL | wx.SL_LABELS, pos=(5, 25))
        self.txt = wx.StaticText(self, label='Sigma', style=wx.ALIGN_CENTER, pos=(80, 5))
        box.Add(self.sigma_sld, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL)
        box.Add((15,15))
        self.SetSizer(box)

        # Rho slider
        self.rho_sld = wx.Slider(self, value=10, minValue=1, maxValue=50,
                                 style=wx.SL_HORIZONTAL| wx.SL_LABELS, pos=(5, 70))
        self.txt = wx.StaticText(self, label='Rho', style=wx.ALIGN_CENTER, pos=(80, 65))
        box.Add(self.rho_sld, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL)
        box.Add((25,15))
        self.SetSizer(box)

        # Vector field density slider
        self.vector_sld = wx.Slider(self, value=20, minValue=1, maxValue=50,
                                    style=wx.SL_HORIZONTAL | wx.SL_LABELS, pos=(5, 120))
        self.txt = wx.StaticText(self, label='Vector Field', style=wx.ALIGN_CENTER, pos=(60, 115))
        box.Add(self.vector_sld, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL)
        box.Add((15,15))
        self.SetSizer(box)

        # Neighbourhood size slider
        self.nhs_sld = wx.Slider(self, value=10, minValue=1, maxValue=50,
                                 style=wx.SL_HORIZONTAL| wx.SL_LABELS, pos=(5, 170))
        self.txt = wx.StaticText(self, label='Neighbourhood', style=wx.ALIGN_CENTER, pos=(50, 170))
        box.Add(self.nhs_sld, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL)
        box.Add((25,15))
        self.SetSizer(box)

        # Select image from file
        self.select_txt = wx.StaticText(self, label='Select Image', pos=(50, 230))
        self.select_file = wx.Button(self, label="Select file", pos=(50, 250))
        self.Bind(wx.EVT_BUTTON, self.SelectFile, self.select_file)

        # Ok Button
        self.ok_but = wx.Button(self, label="OK", pos=(50, 290))
        self.Bind(wx.EVT_BUTTON, self.OkButton, self.ok_but)

        self.Show()

    def SelectFile(self, event):
        """ Open a file"""
        dlg = wx.FileDialog(None, 'choose file')
        if dlg.ShowModal() == wx.ID_OK:
            file_path = dlg.GetPaths()
            # Update class variables
            MyFrame.filepath = file_path
            MyFrame.filename = dlg.GetFilename()

    def OkButton(self, event):
        "When ok button is pressed"
        # Get variables
        sigma = self.sigma_sld.GetValue()
        rho = self.rho_sld.GetValue()
        vector = self.vector_sld.GetValue()
        neighbourhood = self.nhs_sld.GetValue()

        # Update class variables
        MyFrame.sigma = sigma
        MyFrame.rho = rho
        MyFrame.vector = vector
        MyFrame.neighbourhood = neighbourhood

        # Display graphs
        results = Charts()
        results.Show()


class Charts(wx.Frame):
    '''New window to display results'''
    S = defaultdict(dict)
    rows = []
    columns = []

    def __init__(self):
        sigma = str(MyFrame.sigma)
        rho = str(MyFrame.rho)
        vector = str(MyFrame.vector)
        neigh = str(MyFrame.neighbourhood)
        window_name = ('Results _ Sigma=' + sigma +', Rho=' + rho + ', Vector=' + vector +', Neighbourhood=' + neigh)
        wx.Frame.__init__(self, parent=None, title=window_name, size=(650, 600))
        p = wx.Panel(self)
        nb = wx.Notebook(p)

        # Adds tabs to window
        nb.AddPage(Vector_Panel(nb), "Vector Field")
        nb.AddPage(Coherence_Panel(nb), "Coherency")
        nb.AddPage(Gradient_Panel(nb), "Pixel Gradients")
        nb.AddPage(Histogram_Panel(nb), "Histogram")

        # Set notebook in a sizer to create the layout
        sizer = wx.BoxSizer()
        sizer.Add(nb, 1, wx.EXPAND)
        p.SetSizer(sizer)


class Vector_Panel(wx.Panel):

    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent)
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        self.SetSizer(sizer)

        # Get class variables
        sigma = MyFrame.sigma
        rho = MyFrame.rho
        filepath = MyFrame.filepath
        filename = MyFrame.filenmae
        vector = MyFrame.vector

        "Main 2d STRUCTOR TENSOR PROGRAM"
        image = mpimg.imread(filepath[0])
        rows = image.shape[0]  # gets image size
        columns = image.shape[1]
        if len(image.shape) >= 3:  # checks if image is greyscale
            image = rgb2grey(image)

        # Calculate structure tensor & eigenvalues
        S = structure_tensor_2d(image.astype('float'), sigma, rho)
        val, vec = eig_special_2d(S)

        self.axes.imshow(image, cmap=plt.cm.gray)
        plot_orientations(self.axes, image.shape, vec, vector)
        self.axes.set_axis_off()  # Hides tick marks and labels
        self.figure.tight_layout()  # Reduce white space

        # Save results to class variable
        Charts.S = S
        Charts.rows = rows
        Charts.columns = columns


class Coherence_Panel(wx.Panel):

    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent)
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        self.SetSizer(sizer)

        S = Charts.S
        rows = Charts.rows
        columns = Charts.columns
        epsilon = 0.001
        # Calculates coherence
        coherence = np.zeros((rows, columns))
        for i in range(0, rows):
            for j in range(0, columns):
                coherence[i, j] = np.sqrt(
                    ((S[1][i, j] - S[0][i, j]) ** 2 + (4 * S[2][i, j] ** 2)) / (S[1][i, j] + S[0][i, j] + epsilon))
        # Plot coherence
        pos = self.axes.imshow(coherence)
        self.axes.set_axis_off()    # Hides tick marks and labels
        self.figure.tight_layout()  # Reduce white space
        self.figure.colorbar(pos, ax=self.axes)     # Plots colour-bar

class Gradient_Panel(wx.Panel):

    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent)
        self.figure = Figure()
        self.axes = self.figure.subplots(1,2)
        self.canvas = FigureCanvas(self, -1, self.figure)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        self.SetSizer(sizer)

        # Get variables
        S = Charts.S
        # Find min and max of data
        _min, _max = np.amin(S), np.amax(S)
        # Plot Ix data
        self.axes[0].imshow(S[0],vmin = _min, vmax = _max)
        self.axes[0].autoscale(False)
        self.axes[0].set_axis_off()
        self.axes[0].set_title('Ix')
        # Plot Iy data
        pos = self.axes[1].imshow(S[1],vmin = _min, vmax = _max)
        self.axes[1].autoscale(False)
        self.axes[1].set_axis_off()
        self.axes[1].set_title('Iy')
        self.figure.tight_layout()  # Reduce white space

        # Add colourbar for scale
        self.figure.colorbar(pos, ax=self.axes, orientation='horizontal')

class Histogram_Panel(wx.Panel):

    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent)
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()

        S = Charts.S
        rows = Charts.rows
        columns = Charts.columns
        orientation = np.zeros((rows, columns))
        for i in range(0, rows):
            for j in range(0, columns):
                orientation[i, j] = math.degrees(0.5 * math.atan2((2 * S[2][i, j]), (S[1][i, j] - S[0][i, j])))

        flat_ori = orientation.reshape(np.prod(orientation.shape[:2], -1))
        self.axes.hist(flat_ori, bins=range(-90, 90, 1), density=True)
        self.axes.set_xlim((-90, 90))
        self.axes.set_xlabel(u'Vector Orientation (\N{DEGREE SIGN})', fontsize=12, labelpad=10)
        self.axes.set_ylabel('Frequency', fontsize=12, labelpad=10)
        # self.axes.set_ylim((0,0.62))
'''        
        # Plots order parameter (uncomment if required)
        order_list = []
        neighborhood_radius = MyFrame.neighbourhood
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

        self.axes.hist(order_list, bins=np.arange(-1, 1, 0.01))
'''
if __name__ == '__main__':
    app = wx.App()
    frame = MyFrame()
    app.MainLoop()
