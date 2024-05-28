#   This file is part of infdistbayes
#
#   Copyright (C) 2021-2024 C. Ringeval
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

#!/usr/bin/python
import iobayes as iob
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt


def set_figure_params(dpi=None):

    mpl.rcParams["axes.formatter.use_mathtext"] = True
    
    if dpi is not None:
        mpl.rcParams['figure.dpi'] = dpi
        


def finterpsum(x,flist):
    y = 0.0
    for f,w in flist:
        y = y+f(x)*w

    return y

            

def create_1d_figure(name,distrib,weight,xmin=None,xmax=None,ymax=None,
                     xlabelname=None,ylabelname=None,titlename=None,formatname='png',save=False):


    set_figure_params(dpi=200)
    
    fslabel = 12
    
    npts = 10000

    nd = len(distrib)
    nw = len(weight)

    if (nd != nw):
        raise ValueError("distrib and weight of unequal length!")
    

    i=0
    fys=[]
    xpoints = []
    for p in distrib:
        x = p[0]
        y = p[1]
        w = weight[i]
        xsup = [min(x),max(x)]
        f = sp.interpolate.interp1d(x,y,kind='linear',bounds_error=False,fill_value=0.0)

        fys.append( [f,w] )

        xpoints.append(xsup[0])
        xpoints.append(xsup[1])
        
        i+=1
    
    
    xpoints = np.sort(xpoints)

    
#    integval,integerr = sp.integrate.quad(finterpsum,min(xpoints),max(xpoints),args=(fys,)
#                                          ,limit=max(50,len(xpoints)),points=xpoints)
#    print("integ= err= ",integval,integerr)
    
    fig, ax0 = plt.subplots()

    if xmin is None:
        xmin = min(xpoints)

    if xmax is None:
        xmax = max(xpoints)

    if ymax is not None:
        ax0.set_ylim(ymax=ymax)
    
    ax0.set_xlim(xmin,xmax)
    
    ax0.set_xlabel(xlabelname, fontsize=fslabel)
    ax0.set_ylabel(ylabelname,fontsize=fslabel)
    ax0.set_title(titlename,fontsize=fslabel)
    
    xgrid = np.linspace(xmin,xmax,npts)
    ywgrid = finterpsum(xgrid,fys)

    plt.plot(xgrid,ywgrid,color='r',linewidth=0.5)
    plt.ylim(bottom=0.0)

#setting bbox for postscript messes the colorbar        
    if (formatname == 'eps' or formatname == 'ps'):
        plt.savefig(name + '.' +formatname, format=formatname)
    else:
        plt.savefig(name + '.' +formatname, format=formatname, bbox_inches='tight')


    if save:
        minf=-0.01
        pinf = 0.01
        nsave = 100000
        xsave = np.linspace(max(minf,min(xpoints)),min(pinf,max(xpoints)),nsave)
        ywsave = finterpsum(xsave,fys)
        iob.save_probability_1d(name + '.dat',xsave,ywsave)
