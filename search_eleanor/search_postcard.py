import numpy as np
import pyoorb as oo

import astropy
from astropy.io import fits
import numpy as np
import textwrap
import matplotlib.pyplot as plt
import os, sys
import importlib
import glob
from tqdm import tqdm

from astropy.time import Time
from astropy.units import allclose as quantity_allclose
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates.builtin_frames import FK5, ICRS, GCRS, GeocentricMeanEcliptic, BarycentricMeanEcliptic, \
    HeliocentricMeanEcliptic, GeocentricTrueEcliptic, BarycentricTrueEcliptic, HeliocentricTrueEcliptic, \
    HeliocentricEclipticIAU76
from astropy.constants import R_sun, R_earth
from astropy.stats import sigma_clip
from astropy.wcs import WCS

import matplotlib.gridspec as gridspec

from tqdm import tnrange
from tess_stars2px import tess_stars2px_function_entry

from scipy.optimize import minimize
from numpy.linalg import norm

from time import time


class Postcard(object):
    def __init__(self, name, dir='.'):
        self.fn = dir + '/' + name
        print(self.fn)
        self.load_files()
        self.calc_shifts()
        self.make_data()

        self.data.close()
        self.bkg.close()

        self.find_candidates()

    def calc_shifts(self):
        oo.pyoorb.oorb_init()

        def sim_orbit(x):
            x /= norm(x)
            x *= 50

            orbits = np.array([[0, x[0], x[1], x[2], 0.00000, 0.00000,
                                0.00000, 1, self.time[0] + 57000, 1, 12.5, 0.15]],
                              dtype=np.double, order='F')

            mjds = np.array([self.time[0] + 57000])
            epochs = np.array(list(zip(mjds, [1] * len(mjds))), dtype=np.double, order='F')
            eph, err = oo.pyoorb.oorb_ephemeris_full(in_orbits=orbits, in_obscode='C57', in_date_ephems=epochs,
                                                     in_dynmodel='N')

            resid = eph[0][0][1:3] - np.array([self.hdr['CEN_RA'], self.hdr['CEN_DEC']])
            logl = np.sum(resid ** 2)
            return logl

        def get_ephem(x, time):
            x /= norm(x)
            x *= 50

            orbits = np.array([[0, x[0], x[1], x[2], 0.0000, 0.0000,
                                0.00000, 1, self.time[0] + 57000, 1, 12.5, 0.15]],
                              dtype=np.double, order='F')

            mjds = time + 57000
            epochs = np.array(list(zip(mjds, [1] * len(mjds))), dtype=np.double, order='F')
            eph, err = oo.pyoorb.oorb_ephemeris_full(in_orbits=orbits, in_obscode='C57', in_date_ephems=epochs,
                                                     in_dynmodel='N')
            return eph

        res = minimize(sim_orbit, [20, 40., -10.], method='L-BFGS-B', tol=1e-2)

        short_time = np.linspace(self.time[0], self.time[self.gap - 1], 10)
        short_time = np.append(short_time, np.linspace(self.time[self.gap], self.time[-1], 10))

        eph = get_ephem(res.x, short_time)[0][:, 1:3]

        x_, y_ = [], []

        for i in range(len(short_time)):
            ra = eph[i, 0]
            dec = eph[i, 1]  # -14
            ticid = 8675309
            outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, \
            outColPix, outRowPix, scinfo = tess_stars2px_function_entry(
                ticid, ra, dec, trySector=int(self.sector))

            oCP = outColPix[((outCam == self.camera) & (outCcd == self.chip))]
            oRP = outRowPix[((outCam == self.camera) & (outCcd == self.chip))]

            if len(oCP) > 0:
                x_.append(oCP[0])
                y_.append(oRP[0])
            else:
                x_.append(
                    x_[-1] + np.diff(short_time)[i - 1] * (x_[i - 1] - x_[0]) / (short_time[i - 1] - short_time[0]))
                y_.append(
                    y_[-1] + np.diff(short_time)[i - 1] * (y_[i - 1] - y_[0]) / (short_time[i - 1] - short_time[0]))

        x_ -= x_[0]
        y_ -= y_[0]

        self.shift_x = np.interp(self.time, short_time, x_)
        self.shift_y = np.interp(self.time, short_time, y_)

    def load_files(self):
        self.data = fits.open(self.fn)
        self.bkg = fits.open(self.fn.replace('pc.fits', 'bkg.fits'))
        self.hdr = self.data[1].header
        self.sector = self.hdr['SECTOR']
        self.camera = self.hdr['CAMERA']
        self.chip = self.hdr['CCD']
        self.quality = self.data[1].data['quality']
        q = self.quality == 0
        self.time = 0.5 * (self.data[1].data['tstart'] + self.data[1].data['tstop'])[q]
        self.gap = np.where(np.diff(self.time) == np.max(np.diff(self.time)))[0][0] + 1

    def make_data(self):

        q = self.quality == 0

        bkgval = np.nanmedian(self.data[2].data[q], axis=(1, 2))

        bkg1 = sigma_clip(bkgval[0:self.gap], sigma=3, masked=True)
        bkg2 = sigma_clip(bkgval[self.gap:], sigma=3, masked=True)

        self.fmask = np.append(~bkg1.mask, ~bkg2.mask)

        self.size = len(self.data[2].data[q])

        delt = 0.0210

        self.medsub = np.zeros((self.size, 104, 148))

        self.postcard = self.data[2].data[q] - self.bkg[1].data[q]

        for kk in range(0, self.gap):

            if self.fmask[kk] == True:

                tkk = self.time[kk]

                s1 = np.max([tkk - delt * 120, self.time[0]])
                e1 = np.min([tkk - delt * 50, self.time[self.gap]])

                s2 = np.max([tkk + delt * 50, self.time[0]])
                e2 = np.min([tkk + delt * 120, self.time[self.gap]])

                if s1 - 0.3 < e1:
                    if s2 + 0.8 < e2:
                        goodarr = np.where((((self.time > s1) & (self.time < e1)) |
                                            ((self.time > s2) & (self.time < e2))) & (self.fmask == True))[0]

                        pc = self.postcard[goodarr]

                        self.medsub[kk] = self.postcard[kk] - np.nanmean(pc, axis=(0))

        for kk in range(self.gap, self.size):

            if self.fmask[kk] == True:

                tkk = self.time[kk]

                s1 = np.max([tkk - delt * 120, self.time[self.gap]])
                e1 = np.min([tkk - delt * 50, self.time[-1]])

                s2 = np.max([tkk + delt * 50, self.time[self.gap]])
                e2 = np.min([tkk + delt * 120, self.time[-1]])

                if s1 - 0.3 < e1:
                    if s2 + 0.8 < e2:
                        goodarr = np.where((((self.time > s1) & (self.time < e1)) |
                                            ((self.time > s2) & (self.time < e2))) & (self.fmask == True))[0]

                        pc = self.postcard[goodarr]

                        self.medsub[kk] = self.postcard[kk] - np.nanmean(pc, axis=(0))

        self.med = np.nanmedian(self.medsub[0:self.gap][~bkg1.mask], axis=(0))

        self.g1 = np.ma.array(self.med, mask=np.nanstd(self.medsub[0:self.gap][~bkg1.mask], axis=(0)) < 0.40)
        self.g2 = np.ma.array(self.med, mask=np.nanstd(self.medsub[self.gap:][~bkg2.mask], axis=(0)) < 0.40)

        self.medsub_mask = self.medsub + 0.0

        for i in range(self.size):
            if self.fmask[i] == 0:
                self.medsub_mask[i] = 0.0

    def find_candidates(self):

        def do_math(step, max_x, max_y, shiftzero_x, shiftzero_y):
            t1 = time()
            output = np.zeros((self.size, 104 + max_y + 4, 148 + max_x + 4))
            t2 = time()

            shifty = [-1 * np.int(np.round(self.shift_y[i] * step)) for i in range(0, self.size)]
            shiftx = [-1 * np.int(np.round(self.shift_x[i] * step)) for i in range(0, self.size)]

            for i in range(0, self.gap):
                output[i, 2 + shifty[i] + shiftzero_y: 2 + shifty[i] + 104 + shiftzero_y,
                2 + shiftx[i] + shiftzero_x: 2 + shiftx[i] + 148 + shiftzero_x] = self.medsub_mask[i] * self.g1.mask

            t3 = time()
            for i in range(self.gap, self.size):
                output[i, 2 + shifty[i] + shiftzero_y: 2 + shifty[i] + 104 + shiftzero_y,
                2 + shiftx[i] + shiftzero_x: 2 + shiftx[i] + 148 + shiftzero_x] = self.medsub_mask[i] * self.g2.mask
            t4 = time()
            outvar = np.nansum(output, axis=(0))

            std = np.nanstd(outvar[2 + shifty[-1] + shiftzero_y: 2 + shifty[-1] + 104 + shiftzero_y,
                            2 + shiftx[-1] + shiftzero_x: 2 + shiftx[-1] + 148 + shiftzero_x])

            t5 = time()
            if t5 - t1 > 4.0:
                print(np.round(t2-t1, 2), np.round(t3-t2, 2), np.round(t4-t3, 2), np.round(t5-t4, 2))

            if (step > 1e-10):

                loop = True
                while loop == True:

                    best = np.where(outvar == np.max(outvar))
                    summed = outvar[best[0][0] - 1:best[0][0] + 2, best[1][0] - 1:best[1][0] + 2]
                    maxval = np.sum(summed)
                    signif = maxval / std / np.sqrt(9)

                    lightcurve = np.sum(output[:, best[0][0] - 1:best[0][0] + 2,
                                        best[1][0] - 1:best[1][0] + 2], axis=(1, 2))

                    if signif > 7.1:

                        trim = lightcurve[lightcurve != 0.0]
                        sc = sigma_clip(trim, 2.6)

                        signif = np.sum(sc) / std / np.sqrt(9)

                        if signif < 7.1:
                            output[:, best[0][0] - 1:best[0][0] + 2,
                            best[1][0] - 1:best[1][0] + 2] = 0.0
                            outvar = np.nansum(output, axis=(0))

                        else:
                            loop = False
                            xy = WCS(self.hdr, naxis=2).all_pix2world(best[1][0] - shiftzero_x, best[0][0] - shiftzero_y, 2)

                            if shiftx[-1] < 0:
                                xlim0 = shiftzero_x + shiftx[-1]
                                xlim1 = shiftzero_x + 148
                            else:
                                xlim0 = shiftzero_x
                                xlim1 = shiftzero_x + shiftx[-1] + 148

                            if shifty[-1] < 0:
                                ylim0 = shiftzero_y + shifty[-1]
                                ylim1 = shiftzero_y + 104
                            else:
                                ylim0 = shiftzero_y
                                ylim1 = shiftzero_y + shifty[-1] + 104

                            if best[0][0] - ylim0 > 6 and best[0][0] - ylim1 < ylim1 - ylim0 - 7:
                                if best[1][0] - xlim0 > 6 and best[1][0] - xlim1 < xlim1 - xlim0 - 7:
                                    delt = self.time[-1] - self.time[0]
                                    step_x = np.max(shiftx) / delt
                                    step_y = np.max(shifty) / delt

                                    fn_out = 'candidates/{0}-{1}-{2}-{3:04d}-{4:04d}-{5:.1f}-{6:.2f}.png'.format(
                                        self.hdr['SECTOR'],
                                        self.hdr['CAMERA'],
                                        self.hdr['CCD'],
                                        int(self.hdr['CEN_X']),
                                        int(self.hdr['CEN_Y']),
                                        50 / step,
                                        signif)

                                    dt = Time(self.time[0] + 2457000, format='jd').to_datetime()
                                    cd = SkyCoord(ra=xy[0]*u.deg, dec=xy[1]*u.deg, frame='icrs')
                                    rastr = cd.ra.to_string(unit=u.hourangle, sep=' ', precision=2, pad=True)
                                    decstr = cd.dec.to_string(unit=u.deg, sep=' ', precision=2, pad=True)

                                    while True:
                                        try:
                                            plt.close('all')
                                            fig = plt.figure(figsize=(16, 10), constrained_layout=True)
                                            spec = gridspec.GridSpec(ncols=4, nrows=10, figure=fig)
                                            f2 = fig.add_subplot(spec[0:7, 0:])
                                            plt.imshow(outvar[ylim0:ylim1, xlim0:xlim1], origin='lower', vmax=10 * std, vmin=0.0)
                                            plt.colorbar()
                                            f3 = fig.add_subplot(spec[7:9, 0:1])
                                            plt.imshow(outvar[best[0][0] - 6:best[0][0] + 7, best[1][0] - 6:best[1][0] + 7],
                                                       origin='lower')
                                            plt.colorbar()
                                            f2 = fig.add_subplot(spec[7:9, 1:])
                                            plt.plot(self.time[self.fmask],
                                                     np.sum(output[:, best[0][0] - 2:best[0][0] + 3, best[1][0] - 2:best[1][0] + 3],
                                                            axis=(1, 2))[self.fmask])

                                            f3.axis('off')
                                            f3.text(0, -2.7, 'Inferred distance is {0:1f} AU'.format(50 / step))
                                            f3.text(0, -4, 'x shift is {0:.2f} arcsec/day'.format(step_x * 21))
                                            f3.text(0, -5.3, 'y shift is {0:.2f} arcsec/day'.format(step_y * 21))
                                            f3.text(0, -6.6, 'Detection is {0:.1f} counts; {1:.2f} sigma'.format(maxval, signif))
                                            f3.text(0, -7.9,
                                                    'Target at {0:.6f}, {1:.6f} at time {2:3f}'.format(xy[0], xy[1], self.time[0]))
                                            f3.text(0, -9.2,
                                                    'Target at {0}, {1} at time {2} {3} {4}.{5}'.format(rastr, decstr,
                                                                                                        dt.year,
                                                                                                        dt.month,
                                                                                                        dt.day, str(
                                                            np.round(dt.hour / 24, 2))[2:]))
                                            f3.text(0, -10.5,
                                                    'Pixel Coordinates {0}, {1}'.format(best[1][0] - xlim0, best[0][0] - ylim0))
                                            f3.text(0, -11.8, 'Field observed Sector {0}, Camera {1}, Chip {2}'
                                                    .format(self.hdr['SECTOR'], self.hdr['CAMERA'], self.hdr['CCD']))
                                            f3.text(0, -13.1,
                                                    'Postcard {0}, {1}'.format(int(self.hdr['CEN_X']), int(self.hdr['CEN_Y'])))

                                            print(fn_out)
                                            plt.savefig(fn_out, dpi=150)
                                            plt.close(fig)
                                            break
                                        except:
                                            pass


                                    f = open(outfile, 'a')
                                    f.write(
                                        '{0}, {1}, {2}, {3}, {4}, {5:.1f}, {6:.2f}, {7:.2f}, {8:.6f}, {9:.6f}, {10}, {11}, {12:3f} \n'
                                            .format(self.hdr['SECTOR'], self.hdr['CAMERA'], self.hdr['CCD'],
                                                    int(self.hdr['CEN_X']),
                                                    int(self.hdr['CEN_Y']), 50 / step, maxval, signif, xy[0], xy[1],
                                                    best[1][0] - xlim0, best[0][0] - ylim0, self.time[0]))

                                    f.close()
                    else:
                        loop = False

        outfile = 'candidates/report.txt'
        # f = open(outfile, 'w')
        # f.close()

        steps = np.linspace(0.000, 1.6, 75)

        max_x = np.abs(np.int(np.round(np.max(self.shift_x*np.max(steps))))-
                       np.int(np.round(np.min(self.shift_x*np.max(steps)))))
        max_y = np.abs(np.int(np.round(np.max(self.shift_y*np.max(steps))))-
                       np.int(np.round(np.min(self.shift_y*np.max(steps)))))

        shiftzero_x = np.int(np.round(np.max(self.shift_x)/(np.max(self.shift_x) - np.min(self.shift_x)) * max_x))
        shiftzero_y = np.int(np.round(np.max(self.shift_y)/(np.max(self.shift_y) - np.min(self.shift_y)) * max_y))

        for j, step in enumerate(steps):

            do_math(step, max_x, max_y, shiftzero_x, shiftzero_y)




if __name__ == '__main__':
    Postcard(
        dir='postcards',
        name='hlsp_eleanor_tess_ffi_postcard-s0005-1-4-cal-1588-1078_tess_v2_pc.fits')
