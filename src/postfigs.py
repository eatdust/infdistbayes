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
import getdist.densities as gd

def set_figure_params(dpi=None):

    mpl.rcParams["axes.formatter.use_mathtext"] = True
    
    if dpi is not None:
        mpl.rcParams['figure.dpi'] = dpi
        


def finterpsum(x,flist):
    y = 0.0
    for f,w in flist:
        y = y+f(x)*w

    return y


def finterpsum_2d(x,y,flist):
    z = 0.0
    for f,w in flist:
        z = z+f(x,y)*w

    return z

            

def create_1d_figure(name,distrib,weight,xmin=None,xmax=None,ymax=None,
                     xlabelname=None,ylabelname=None,titlename=None,formatname='png',
                     npts=10000,save=False,nsave=100000):


    set_figure_params(dpi=200)
    
    fslabel = 12
    
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
        xsave = np.linspace(xmin,xmax,nsave)
        ywsave = finterpsum(xsave,fys)
        iob.save_probability_1d(name + '.dat',xsave,ywsave)



def create_2d_figure(name,distrib2D,weight,xmin=None,xmax=None,ymin=None, ymax=None,
                     xlabelname=None,ylabelname=None,clabelname=None,titlename=None,
                     formatname='png',npts=100,nlevels=100,clevels=[0.98,0.95,0.68]):


    set_figure_params(dpi=200)
    
    fslabel = 12
    
    nd = len(distrib2D)
    nw = len(weight)

    if (nd != nw):
        raise ValueError("distrib and weight of unequal length!")
    

    i=0
    fPs=[]
    xpoints = []
    ypoints= []
    for p in distrib2D:
        x = p[0]
        y = p[1]
        z = p[2]
        w = weight[i]
        xsup = [min(x),max(x)]
        ysup = [min(y),max(y)]
        
        f = sp.interpolate.RectBivariateSpline(x,y,z,kx=1,ky=1)
        
        fPs.append( [f,w] )

        xpoints.append(xsup[0])
        xpoints.append(xsup[1])
        ypoints.append(ysup[0])
        ypoints.append(ysup[1])
        
        i+=1
    
    
    xpoints = np.sort(xpoints)
    ypoints = np.sort(ypoints)
    
    fig, ax0 = plt.subplots()


    #super-slow crazy consistency check, this should be unity. Mind that dblquad integrate f(y,x)...
#    integval,integerr = sp.integrate.dblquad(finterpsum_2d,min(ypoints),max(ypoints),
#                                             min(xpoints),max(xpoints), args=(fPs,))
#    print("integ= err= ",integval,integerr)

    

    if xmin is None:
        xmin = min(xpoints)

    if xmax is None:
        xmax = max(xpoints)

    if ymin is None:
        ymin = min(ypoints)

    if ymax is None:
        ymax = max(ypoints)        
    
    ax0.set_xlabel(xlabelname, fontsize=fslabel)
    ax0.set_ylabel(ylabelname,fontsize=fslabel)
    ax0.set_title(titlename,fontsize=fslabel)
    
    xgrid = np.linspace(xmin,xmax,npts)
    ygrid = np.linspace(ymin,ymax,npts)
    Pwgrid = finterpsum_2d(xgrid,ygrid,fPs)

        
#    print("xgrid= ",xgrid)
#    print("ygrid= ",ygrid)
#    print("Pwgrid= ",Pwgrid)

    gdP2D = gd.Density2D(xgrid,ygrid,Pwgrid)
    gdlevels = gdP2D.getContourLevels(contours=clevels)
#    print("contour levels ",gdlevels)
#    print("should be unity ",gdP2D.integrate(Pwgrid))

    cf = ax0.contourf(xgrid,ygrid,Pwgrid,levels=nlevels)
    ccl = ax0.contour(xgrid,ygrid,Pwgrid,levels=gdlevels,colors=['grey','lightsalmon','cyan'])

    cbar = fig.colorbar(cf)
    cbar.ax.set_ylabel(clabelname)
    cbar.add_lines(ccl)

    
#setting bbox for postscript messes the colorbar        
    if (formatname == 'eps' or formatname == 'ps'):
        plt.savefig(name + '.' +formatname, format=formatname)
    else:
        plt.savefig(name + '.' +formatname, format=formatname, bbox_inches='tight')








        
