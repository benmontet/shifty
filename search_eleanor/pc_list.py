import numpy as np

a = np.loadtxt('/Users/bmontet/research/tess/eleanor/eleanor/postcard_centers.txt', dtype='int', skiprows=1)

sec = '0005'
camera = '2'
chip = '3'

outfile = open('wget_{0}-{1}-{2}.sh'.format(sec, camera, chip), 'w')

dir = 'http://archive.stsci.edu/hlsps/eleanor/postcards/s{0}/{1}-{2}/'.format(sec, camera, chip)

for i in range(len(a)):
    fn = 'hlsp_eleanor_tess_ffi_postcard-s{0}-{1}-{2}-cal-{3:04d}-{4:04d}_tess_v2_pc.fits'.format(
        sec, camera, chip, a[i,0], a[i,1])

    outfile.write('wget {0}{1} \n'.format(dir, fn))
    outfile.write('wget {0}{1} \n'.format(dir, fn.replace('pc.fits', 'bkg.fits')))
outfile.close()