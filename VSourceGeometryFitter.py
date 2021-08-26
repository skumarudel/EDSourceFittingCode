__author__ = 'Sajan Kumar'

# These are the libraries you need for doing the source fitting analysis
# Essentially you need root pyroot and Sherpa package
import numpy as np
from matplotlib import pyplot as plt
from sherpa.fit import Fit
from sherpa import models
from sherpa.data import Data1D, Data2D
from sherpa.stats import Cash, CStat
from sherpa.estmethods import Confidence
from sherpa import optmethods
import sys
sys.path.append("/Users/kumar/research/software/root_install/root_build_6.14.00/lib")
import ROOT
import matplotlib.lines
import matplotlib.patches as patches
import sherpa.astro.models as md
from MathFunction import *
from sherpa.plot import RegionProjection
from sherpa.plot import IntervalProjection


# --------------------------------------------------------------

class VSourcePositionFitting(object):
    def __init__(self, dir, filename):
        self.datafile = str(dir) + "/" + str(filename)

    def readEDFile(self):
        self.Rfile = ROOT.TFile(self.datafile, "read")
        tRunsummary = self.Rfile.Get("total_1/stereo/tRunSummary")

        crval1, crval2 = [], []
        for i, events in enumerate(tRunsummary):
            crval1.append(events.SkyMapCentreRAJ2000)
            crval2.append(events.SkyMapCentreDecJ2000)
        self.Ra = crval1[0]
        self.Dec = crval2[0]
        # return self.Ra, self.Dec

    def extract_data(self, hmap, deg=0.3):
        xbinlow = hmap.GetXaxis().FindBin(-deg)
        xbinhigh = hmap.GetXaxis().FindBin(deg)
        ybinlow = hmap.GetYaxis().FindBin(-deg)
        ybinhigh = hmap.GetYaxis().FindBin(deg)
        print(xbinlow, xbinhigh, ybinlow, ybinhigh)
        xrange = xbinhigh - xbinlow + 1
        yrange = ybinhigh - ybinlow + 1
        z = np.zeros((xrange, yrange))
        for i in range(xrange):
            for j in range(yrange):
                z[i, j] = hmap.GetBinContent(ybinlow + j, xbinlow + i)
        return z

    def get_maps(self, deg=0.3):
        self.deg = deg
        OnMap = "total_1/stereo/skyHistograms/hmap_stereoUC_on"
        hmap_UC_on = self.Rfile.Get(OnMap)
        OffMap = "total_1/stereo/skyHistograms/hmap_stereoUC_off"
        hmap_UC_off = self.Rfile.Get(OffMap)
        alphaMap = "total_1/stereo/skyHistograms/hmap_alphaNormUC_off"
        hmap_UC_alpha = self.Rfile.Get(alphaMap)
        self.On = self.extract_data(hmap_UC_on, deg)
        self.Off = self.extract_data(hmap_UC_off, deg)
        self.alpha = self.extract_data(hmap_UC_alpha, deg)
        self.NormOff = self.Off * self.alpha
        return self.On

    def fit(self, xpos, ypos, gaus=True):
        self.Model = gaus
        binsize = 0.05
        x = np.arange(-self.deg, self.deg + binsize, binsize)
        y = np.arange(-self.deg, self.deg + binsize, binsize)
        x_range, y_range = np.meshgrid(x, y)
        #Use a constant 2D function to fit the background on Normalized Off map and then freeze this parameter
        # when fit the Gauss or King function on On data
        bg = models.Const2D('bg')

        d_b = Data2D('bg', x_range.flatten(), y_range.flatten(), self.NormOff.flatten(), shape=x_range.shape)
        #
        # print(bg)
        #
        f_b = Fit(d_b, bg, CStat(), optmethods.NelderMead())
        res_bg = f_b.fit()
        bg.c0.freeze()
        # print(bg.c0)

        if gaus:
            source = models.Gauss2D('source')
            source.fwhm = 0.08
            source.ampl = 100
            source.ampl.min = 0
            source.ampl.max = 500
            # source.fwhm.min = 0
            source.fwhm.max = 0.5
            source.xpos.min = -self.deg
            source.xpos.max = self.deg
            source.ypos.min = -self.deg
            source.ypos.max = self.deg

            source.xpos.val = xpos
            source.ypos.val = ypos
            OnSourceModel = source + bg

            d_s = Data2D('signal', x_range.flatten(), y_range.flatten(), self.On.flatten(), shape=x_range.shape)
            f_s = Fit(d_s, OnSourceModel, CStat(), optmethods.NelderMead())

            res2 = f_s.fit()
            print(res2.format())
            f_s.method = optmethods.NelderMead()
            f_s.estmethod = Confidence()
            err_s = f_s.est_errors()
            param = predict(err_s)
            # print(param)
            print(err_s.format())
            self.containmentR, self.containmentRerr = convertFWHMto68(param['fwhm'][0]), convertFWHMto68(
                param['fwhm'][1])
            self.x, self.y, self.xerr, self.yerr = param['xpos'][0], param['ypos'][0], \
                                                   param['xpos'][1], param['ypos'][1]
            convert_derotated_RADECJ2000(self.Ra, self.Dec, self.x, self.y, self.xerr, self.yerr)

            #       print("Source sigma (PSF)")

            print("68% Containment radius = {:.3f} +/- {:.3f} degrees".format(self.containmentR, self.containmentRerr))
            print("----Containment radius in ArcMinute----\n")
            print('68% Containment radius= {:.1f} +/- {:.1f}'.format(self.containmentR * 60, self.containmentRerr * 60))
        #        print (str(convertFWHMto68(err_s.parvals[0])) + '+/-'+ str(convertFWHMto68(err_s.parmaxes[0])))
        else:
            beta = md.Beta2D('beta')
            beta.r0 = 0.04
            beta.alpha = 1.95
            beta.xpos.min = -self.deg
            beta.xpos.max = self.deg
            beta.ypos.min = -self.deg
            beta.ypos.max = self.deg
            beta.r0.max = 0.3
            OnSourceModel = beta + bg
            beta.alpha.freeze()

            d_s = Data2D('signal', x_range.flatten(), y_range.flatten(), self.On.flatten(), shape=x_range.shape)
            f_s = Fit(d_s, OnSourceModel, CStat(), optmethods.NelderMead())

            res2 = f_s.fit()
            print(res2.format())
            f_s.method = optmethods.NelderMead()
            f_s.estmethod = Confidence()
            err_s = f_s.est_errors()
            param = predict(err_s)
            print(err_s.format())
            
            self.x, self.y, self.xerr, self.yerr = param['xpos'][0], param['ypos'][0], \
                                                   param['xpos'][1], param['ypos'][1]
            convert_derotated_RADECJ2000(self.Ra, self.Dec, self.x, self.y, self.xerr, self.yerr)
            self.r0 = param['r0'][0]
            print("Core radius (r0) = {:.3f} +/- {:.3f}".format(param['r0'][0], param['r0'][1]))
            print("Alpha = {:.3f} +/- {:.3f}".format(param['alpha'][0], param['alpha'][1]))
            Crad = containment_radius(param['r0'][0], param['alpha'][0])
            CradH = containment_radius(param['r0'][0] + param['r0'][1], param['alpha'][0] + param['alpha'][1])
            # CradL = containment_radius(param['r0'][0] - param['r0'][1], param['alpha'][0] - param['alpha'][1])
            Craderr = abs(CradH - Crad)
            print("68% Containment radius = {:.3f} +/- {:.3f}".format(Crad, Craderr))
            '''
            intproj = IntervalProjection()
            intproj.calc(f_s, beta.r0)
            intproj.plot()
            plt.show()
            
            regproj = RegionProjection()
            # regproj.prepare(min=[1, 0], max=[5, 0.15], nloop=(21, 21))
            regproj.calc(f_s, beta.alpha, beta.r0)
            regproj.contour()
            plt.show()
            '''
            # print("Alpha = {:.3f} +/- {:.3f})".format(param['alpha'][0], param['alpha'][1]))

    def plot_skymap(self):

        plt.imshow(self.On, cmap='rainbow', origin="lower", extent=[-self.deg, self.deg, -self.deg, self.deg])

        plt.plot(self.x - self.xerr, self.y, self.x + self.xerr, self.y, linewidth=2)
        plt.colorbar()
        plt.xlabel("X position on Sky")
        plt.ylabel("Y position on Sky")
        axes = plt.gca()
        line1 = matplotlib.lines.Line2D((self.x - self.xerr, self.x + self.xerr), (self.y, self.y), color='black')
        line2 = matplotlib.lines.Line2D((self.x, self.x), (self.y - self.yerr, self.y + self.yerr), color='black')
        axes.add_line(line1)
        axes.add_line(line2)
        if self.Model:
            circle = patches.Circle((self.x, self.y), self.containmentR, color='black', fill=False)
        else:
            circle = patches.Circle((self.x, self.y), self.r0, color='black', fill=False)
        axes.add_patch(circle)
        plt.show()

