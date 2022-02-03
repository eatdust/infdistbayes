#   This file is part of infdistbayes
#
#   Copyright (C) 2021 C. Ringeval
#   
#   infdistbayes is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   infdistbayes is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with infdistbayes.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from scipy import integrate

def load_bayesdist(filename):
    disttype = np.dtype([('Name', np.unicode_, 60),
                     ('Nparams',np.dtype(int)),
                     ('Evidence',np.dtype(float)),
                     ('Error',np.dtype(float)),
                     ('Complexity',np.dtype(float)),
                     ('NparamsMinusComp',np.dtype(float)),
                     ('Best',np.dtype(float))])


    
    bayesdist = np.loadtxt(filename,dtype=disttype)

    if (bayesdist.ndim == 0):
        bayesdist = np.reshape(bayesdist,1)
        
        
    return bayesdist



def load_margestats(filename,**kwargs):
    margetype = np.dtype([('parameter', np.unicode_, 60),
                          ('mean', np.dtype(float)),
                          ('sddev', np.dtype(float)),
                          ('lower1', np.dtype(float)),
                          ('upper1', np.dtype(float)),
                          ('limit1', np.unicode_, 60),
                          ('lower2', np.dtype(float)),
                          ('upper2', np.dtype(float)),
                          ('limit2', np.unicode_, 60),
                          ('texform',np.unicode_, 60)])

    print("filename is ",filename)
    
    if kwargs:
        margestats = np.loadtxt(filename,margetype,usecols=(0,1,2,3,4,5,6,7,8,9),**kwargs)
    else:
        margestats = np.loadtxt(filename,margetype,usecols=(0,1,2,3,4,5,6,7,8,9),skiprows=3)

    return margestats



def load_likestats(filename,**kwargs):
    liketype = np.dtype([('parameter', np.unicode_, 60),
                         ('bestfit', np.dtype(float)), \
                         ('lower1', np.dtype(float)),
                         ('upper1', np.dtype(float)),
                         ('lower2', np.dtype(float)),
                         ('upper2', np.dtype(float))])
    if kwargs:
        likestats = np.loadtxt(filename,liketype,usecols=(0,1,2,3,4,5),**kwargs)
    else:
        likestats = np.loadtxt(filename,liketype,usecols=(0,1,2,3,4,5),skiprows=6)              

    return likestats


def load_and_normalize_probability(filename):
    x,y = np.loadtxt(filename,unpack=True,usecols=[0,1])
    norm = integrate.simps(y,x)
    return x,y/norm


def save_probability_1d(filename,x,P):
    todump = np.column_stack([x,P])
    np.savetxt(filename,todump)
