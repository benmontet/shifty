import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import glob
from multiprocessing.dummy import Pool
import tqdm
import search_postcard as sp
from time import time

dir = '/data/quokka/bmontet/postcards/s0005/2-2/'
flist = glob.glob('{0}/*_pc.fits'.format(dir))
#pbar = tqdm.tqdm(total=len(flist))
counter = 0

def run_postcard(fname):
    global counter
    fn = fname.split('/')[-1]
    sp.Postcard(dir=dir,name=fn)
    tnow = time()
    counter += 1
    print(counter, np.round(tnow-t1, 2), np.round(len(flist)/counter * (tnow-t1), 2))

t1 = time()

pool = Pool(6)
#for _ in tqdm.tqdm(pool.apply_async(run_postcard, flist), total=len(flist)):
#    pass

results = [pool.apply_async(run_postcard, args=[flist[i]]) for i in range(len(flist))]
[res.get() for i, res in enumerate(results)]

#res = [pool.apply_async(run_postcard, flist) for i in range(6)]
