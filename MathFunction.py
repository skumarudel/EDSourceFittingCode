import numpy as np
from pyslalib import slalib
from astropy import units as u
from astropy.coordinates import SkyCoord
import math
from sherpa.models import Gauss2D
from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from scipy import optimize, integrate


def simulate_data(hmap, deg=0.3):
    xbinlow = hmap.GetXaxis().FindBin(-deg)
    xbinhigh = hmap.GetXaxis().FindBin(deg)
    ybinlow = hmap.GetYaxis().FindBin(-deg)
    ybinhigh = hmap.GetYaxis().FindBin(deg)
    # print(xbinlow, xbinhigh, ybinlow, ybinhigh)
    xrange = xbinhigh - xbinlow + 1
    yrange = ybinhigh - ybinlow + 1
    z = np.zeros((xrange, yrange))
    for i in range(xrange):
        for j in range(yrange):
            z[i, j] = np.random.poisson(hmap.GetBinContent(ybinlow + j, xbinlow + i))
    return z


def simulate2DGauss(amplitude,xpos,ypos,sigma, deg):
    binsize = 0.05
    x = np.arange(-deg, deg + binsize, binsize)
    y = np.arange(-deg, deg + binsize, binsize)
    x_range, y_range = np.meshgrid(x, y)
    src = Gauss2D('src')
    model = src
    fwhm = sigma * gaussian_sigma_to_fwhm
    src.xpos, src.ypos, src.fwhm, src.ampl = xpos, ypos, fwhm, amplitude
    mexp = model(x_range.flatten(), y_range.flatten())
    mexp.resize(x_range.shape)
    mexp.shape
    msim = np.random.poisson(mexp)
    return msim


def deg2rad(deg):
    return deg / 180. * np.pi


def rad2deg(rad):
    return rad * 180. / np.pi


def convertFWHMto68(value):
    sigma = value / 2.355
    return 1.51*sigma


def convert_derotated_RADECJ2000(SourceRa, SourceDec, x_deg, y_deg, x_err, y_err):
    print("Sky map centre: " +  str(SourceRa) + " " + str(SourceDec))
    ra,dec = slalib.sla_dtp2s(deg2rad(x_deg),deg2rad(y_deg),deg2rad(SourceRa),deg2rad(SourceDec))
    ra_err,dec_err = slalib.sla_dtp2s(deg2rad(x_deg+x_err),deg2rad(y_deg+y_err),deg2rad(SourceRa),deg2rad(SourceDec))
    ra_deg = rad2deg(ra)
    dec_deg = rad2deg(dec)
    ra_err_deg = rad2deg(ra_err)
    dec_err_deg = rad2deg(dec_err)
    ra_err_deg  = abs( ra_deg - ra_err_deg );
    dec_err_deg = abs( dec_deg - dec_err_deg );
    print("(RA,Dec) (J2000) for (x,y)=({:.3f} +/- {:.3f}, {:.3f} +/- {:.3f})".format(x_deg, x_err, y_deg, y_err))
    print("(RA,Dec) (J2000) for (Ra,Dec)=({:.3f} +/- {:.3f}, {:.3f} +/- {:.3f})".format(ra_deg, ra_err_deg, dec_deg, dec_err_deg))
    c = SkyCoord(ra=ra_deg*u.degree, dec=dec_deg*u.degree, frame='fk5')
    c_err = SkyCoord(ra=ra_err_deg*u.degree, dec=dec_err_deg*u.degree, frame='fk5')
    ra_h, ra_m, ra_s = c.ra.hms
    dec_d, dec_m, dec_s = c.dec.dms
    ra_err_h, ra_err_m, ra_err_s = c_err.ra.hms
    dec_err_d, dec_err_m, dec_err_s = c_err.dec.dms

    print("(RA,Dec) (J2000) for (Ra,Dec)=({}h {}m {:.1f}s +/- {}h {}m {:.1f}s , {}d {}m {:.1f}s +/- {}d {}m {:.1f}s)".format(ra_h,ra_m,ra_s, ra_err_h, ra_err_m, ra_err_s, dec_d,dec_m,dec_s, dec_err_d, dec_err_m, dec_err_s))

    camera_offset = math.sqrt( x_deg * x_deg + y_deg * y_deg )
    print("Offset from camera center = {:.3f} degrees".format(camera_offset))



def predict(err_s):
    value = np.asarray(err_s.parvals)
    name = np.asarray(err_s.parnames)
    parname = np.asarray([x.split('.')[1] for x in name])
    errhigh = np.asarray(err_s.parmaxes)
    errlow = np.abs(np.asarray(err_s.parmins))
    meanerr = (errhigh + errlow) * 0.5
    combined = np.vstack((value, meanerr)).T
    return dict(zip(parname, combined))


def containment_radius(r0, beta):
    r0 = r0
    beta = beta
    amplitude = 100
    func_king = lambda r: amplitude * (1 + (r**2 / r0**2)) ** (-beta)
    root_fn = lambda x: integrate.quad(func_king, -x, x)[0] / integrate.quad(func_king, -5, 5)[0] - 0.68
    value = optimize.brentq(root_fn, 0, 10)
    return value