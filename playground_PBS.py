from re import S
from tkinter import N
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.colors as colors
import numpy as np


xLim = [-1.5, 1.5]
yLim = [-1.5, 1.5]
res = 50

class Interaction(object):
    '''
    Adapted from https://stackoverflow.com/questions/55324129/how-to-plot-interactive-draw-able-image-using-matplotlib-in-google-colab
    '''
    def __init__(self, axes):
        self.axes = axes
        self.corners = []
        self.initAxis()

    def initAxis(self):
        self.axes.clear()
        self.axes.set_axis_off()
        self.axes.set_xlim(xLim[0], xLim[1])
        self.axes.set_ylim(yLim[0], yLim[1])

    def mouse_press(self, event):
        if (event.button == 3) :
            x, y = event.xdata, event.ydata
            if (len(self.corners) == 4) :
                self.corners.clear()
                self.initAxis()
            self.corners.append([x, y])

            self.axes.scatter(x, y, s = 20, facecolors='none', edgecolors='green')
            plt.draw()
            if (len(self.corners) == 4) :
                for ind in range(len(self.corners)) :
                    n_ind = (ind + 1) % 4
                    X = [self.corners[ind][0], self.corners[n_ind][0]]
                    Y = [self.corners[ind][1], self.corners[n_ind][1]]
                    self.axes.plot(X, Y, "--", c = "green")
    
    def loadDefault(self, event):
        self.initAxis()
        self.corners.clear()
        self.corners.append([1.0, 1.0])
        self.corners.append([1.0, 0.0])
        self.corners.append([0.0, 0.0])
        self.corners.append([0.0, 1.0])

        for ind in range(len(self.corners)) :
            n_ind = (ind + 1) % 4
            X = [self.corners[ind][0], self.corners[n_ind][0]]
            Y = [self.corners[ind][1], self.corners[n_ind][1]]
            self.axes.plot(X, Y, "--", c = "green")
            self.axes.scatter(X[0], Y[0], s = 20, facecolors='none', edgecolors='green')
        plt.draw()


def drawQuad(ax, corners, s_ind = None, alpha = 1.0):
    for ind in range(len(corners)) :
        n_ind = (ind + 1) % 4
        X = [corners[ind][0], corners[n_ind][0]]
        Y = [corners[ind][1], corners[n_ind][1]]
        ax.plot(X, Y, "--", c = "green", alpha = alpha)
        if (s_ind is not None) and(ind == s_ind) :
            ax.scatter(X[0], Y[0], s = 40, facecolors='none', edgecolors='blue', alpha = alpha)
        else:
            ax.scatter(X[0], Y[0], s = 20, facecolors='none', edgecolors='green', alpha = alpha)

def mirrored(maxVal, halfN):
    end = np.linspace(0, maxVal, halfN)
    start = list(-end[-1:0:-1])
    return start + list(end)

def main():
    #-----INPUT REST QUAD /!\ ORDER /!\-----
    fig, axes = plt.subplots(figsize=(5, 5))
    inter = Interaction(axes)
    plt.connect('button_press_event', inter.mouse_press)
    axDefaultButton= plt.axes([0.9, 0.0, 0.1, 0.075])
    defaultButton = Button(axDefaultButton, 'Default')
    defaultButton.on_clicked(inter.loadDefault)
    axes.plot()
    plt.show()
    restCorners = np.array(inter.corners.copy())
    if (len(restCorners) != 4):
        return 0

    #-----INPUT DEFORMED QUAD /!\ ORDER /!\-----
    fig, axes = plt.subplots(figsize=(5, 5))
    inter = Interaction(axes)
    plt.connect('button_press_event', inter.mouse_press)
    axDefaultButton= plt.axes([0.9, 0.0, 0.1, 0.075])
    defaultButton = Button(axDefaultButton, 'Default')
    defaultButton.on_clicked(inter.loadDefault)
    drawQuad(axes, restCorners, alpha = 0.25)
    axes.plot()
    plt.show()
    defCorners = inter.corners.copy()
    if (len(defCorners) != 4):
        return 0

    #----COMPUTE BASIS FUNCTIONS-----
    basisCoeffs = np.zeros((4, 4), dtype=float) #A line i correspond to coeff of the i basis function
    A = np.zeros((4, 4), dtype=float)
    for i in range(4):
        A[i, 0] = restCorners[i][0] * restCorners[i][1] 
        A[i, 1] = restCorners[i][0]
        A[i, 2] = restCorners[i][1]
        A[i, 3] = 1.0
    A_inv = np.linalg.inv(A)
    for i in range(4):
        b = np.zeros(4, dtype=float)
        b[i] = 1.0
        basisCoeffs[i] = A_inv.dot(b)
    print("Computing Basis Functions: DONE")
    print(basisCoeffs)


    #-----DISPLAY BASIS FUNCTIONS-----
    x = np.linspace(xLim[0], xLim[1], res)
    y = np.linspace(yLim[0], yLim[1], res)
    xx, yy = np.meshgrid(x, y)
    z = []
    for i in range(4):
        z.append(basisCoeffs[i, 0] * xx * yy + basisCoeffs[i, 1] * xx + basisCoeffs[i, 2] * yy + basisCoeffs[i, 3])

    fig1, ax2 = plt.subplots(2, 2, constrained_layout=True)
    ind = 0
    divnorm = colors.TwoSlopeNorm(vmin = -1.5, vcenter=0., vmax=1.5)
    levels = mirrored(1., 25)
    for i in range(2):
        for j in range(2):
            CS = ax2[i, j].contourf(x, y, z[ind], levels, cmap="bwr", norm = divnorm)
            ax2[i, j].set_aspect('equal', 'box')
            ax2[i, j].set_title(str(ind) + ": " 
            + ("%.2f" % basisCoeffs[ind, 0]) + ", " 
            + ("%.2f" % basisCoeffs[ind, 1]) + ", "
            + ("%.2f" % basisCoeffs[ind, 2]) + ", "
            + ("%.2f" % basisCoeffs[ind, 3]))
            drawQuad(ax2[i, j], restCorners, ind)
            ind += 1
    cbar_ax = fig1.add_axes([0.85, 0.15, 0.05, 0.7])
    fig1.colorbar(CS, cax = cbar_ax)
    fig1.subplots_adjust(right=0.8)
    plt.show()


    #-----CHECK IS THE REST QUAD IS WELL MAP TO THE DEFORMED QUAD USING BASIS FUNCTION-----
    #cf https://stackoverflow.com/questions/71735261/how-can-i-show-transformation-of-coordinate-grid-lines-in-python (avec grid = les 4 cotes du rest quad)
    mapBorders = []
    t = np.linspace(0.0, 1.0, res)
    t = np.expand_dims(t, 1)
    for ind in range(4):
        n_ind = (ind + 1) % 4
        start = np.expand_dims(restCorners[ind], 0)
        end = np.expand_dims(restCorners[n_ind], 0)
        sampledPoints = (1.0 - t) * start + t * end
        W = []
        for i in range(4) :
            W.append(basisCoeffs[i, 0] * sampledPoints[:, 0] * sampledPoints[:, 1]
                 + basisCoeffs[i, 1] * sampledPoints[:, 0]
                 + basisCoeffs[i, 2] * sampledPoints[:, 1] + basisCoeffs[i, 3])
        mapB = np.zeros((res, 2))
        for i in range(4):
            mapB += np.expand_dims(W[i], 1) * np.expand_dims(defCorners[i], 0)
        mapBorders.append(mapB)
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(xLim[0], xLim[1])
    ax.set_ylim(yLim[0], yLim[1])
    drawQuad(ax, defCorners, alpha=0.25)
    for ind in range(4):
        ax.plot(mapBorders[ind][:, 0], mapBorders[ind][:, 1], c = "green")
        #ax.scatter(mapBorders[ind][:, 0], mapBorders[ind][:, 1], s = 5, facecolors='none', edgecolors='green')
    plt.show()










if __name__ == "__main__":
    main()