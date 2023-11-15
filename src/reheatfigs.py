#   This file is part of infdistbayes
#
#   Copyright (C) 2021-2023 C. Ringeval
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
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def set_figure_params(xlabsize,ylabsize,unit):
    mpl.rcParams['xtick.labelsize'] = xlabsize
    mpl.rcParams['ytick.labelsize'] = ylabsize
    mpl.rcParams['mathtext.fontset'] = unit


def create_2d_figure(name,lnxmin,lnxmax,ymin,ymax,cname,formatname,
                     lnxdata,ydata,ydataMean,ydataVar,cdata,
                     xlabelname=None,ylabelname=None,labelname=None):

    set_figure_params(12,12,'cm')

    fig, ax0 = plt.subplots()
    plt.axis([0,1,0,1])

    xmin = np.exp(lnxmin)
    xmax = np.exp(lnxmax)

    fsjeffrey = 8
    fslabel = 12
    fscblabel=14

    #ax0.set_title('Observing the inflationary reheating',fontsize=8)

    ax0.set_xlabel(xlabelname,fontsize=fslabel)
    ax0.set_xscale('log')
    ax0.set_xlim(xmin, xmax)
    ax0.set_ylabel(ylabelname,fontsize=fslabel)
    ax0.set_ylim(ymin, ymax)

    #color scale
    c = plt.scatter(np.exp(lnxdata),ydata,s=100,c=cdata,
                    linewidths=0.5,edgecolors='black',cmap='jet',zorder=10)


    #legends and labels
    totor=(1,0.705,0)
    totob=(0,0.866,1)

    #Jeffrey Scale
    Emax=0.
    Einc = Emax - 1.0
    Eweak = Emax - 2.5
    Emod = Emax - 5.0

    Bmax = np.exp(Emax)
    Binc = np.exp(Einc)
    Bweak = np.exp(Eweak)
    Bmod = np.exp(Emod)


    if formatname == 'eps':
      ax0.fill_between(x=[Binc,xmax],y1=[ymax,ymax],y2=ymin,facecolor='aquamarine', alpha=0.2)
      ax0.fill_between(x=[Bweak,Binc],y1=[ymax,ymax],y2=ymin,facecolor='skyblue', alpha=0.2)
      ax0.fill_between(x=[Bmod,Bweak],y1=[ymax,ymax],y2=ymin,facecolor='darksalmon', alpha=0.2)
      ax0.fill_between(x=[xmin,Bmod],y1=[ymax,ymax],y2=ymin,facecolor='grey', alpha=0.4)
    else:
      #nice colors for png with alpha channel
      ax0.fill_between(x=[Binc,xmax],y1=[ymax,ymax],y2=ymin,facecolor=totob, alpha=0.2)
      ax0.fill_between(x=[Bweak,Binc],y1=[ymax,ymax],y2=ymin,facecolor=totor, alpha=0.2)
      ax0.fill_between(x=[Bmod,Bweak],y1=[ymax,ymax],y2=ymin,facecolor='red', alpha=0.2)
      ax0.fill_between(x=[xmin,Bmod],y1=[ymax,ymax],y2=ymin,facecolor='grey', alpha=0.4)

  
    ytext=ymax + 0.1
    xinc = np.exp(Einc+0.5*(lnxmax-Einc))
    xweak = np.exp(Eweak+0.5*(Einc-Eweak))
    xmod = np.exp(Emod + 0.5*(Eweak-Emod))
    xstg = np.exp(lnxmin + 0.5*(Emod-lnxmin))

    ax0.text(xinc,ytext, 'favored', color='black',fontsize=fsjeffrey
             ,horizontalalignment='center',verticalalignment='center')
    ax0.text(xweak,ytext, 'weakly disfavored', color='black',fontsize=fsjeffrey
             ,horizontalalignment='center',verticalalignment='center')
    ax0.text(xmod,ytext, 'moderately disfavored', color='black',fontsize=fsjeffrey
             ,horizontalalignment='center',verticalalignment='center')
    ax0.text(xstg,ytext, 'strongly disfavored', color='black',fontsize=fsjeffrey
             ,horizontalalignment='center',verticalalignment='center')
    
    if ydataMean is not None:
        ymean = ydataMean
        ystddev= np.sqrt(ydataVar)
        yone = ymean - 0.5*ystddev
        ytwo = ymean + 0.5*ystddev
        ax0.fill_between(x=[xmin,xmax],y1=[yone,yone],y2=[ytwo,ytwo],facecolor='yellow', alpha=0.8)
        
        
    if name == 'plc_2':
      ymeanplc1 = 0.556088498289
      ystddevplc1 = 0.286561281445 
      #  plt.plot((xmin,xmax),(ymeanplc1,ymeanplc1),'k:')
      #  plt.plot((xmin,xmax),(ymeanplc1+ystddevplc1/2,ymeanplc1+ystddevplc1/2),'k:')
      #  plt.plot((xmin,xmax),(ymeanplc1-ystddevplc1/2,ymeanplc1-ystddevplc1/2),'k:')


    #reheating energy colorbar
    cb = fig.colorbar(c,orientation='vertical',pad=0.01,aspect=30,fraction=0.05
                          ,ticks=[-10,-20,-30,-40,0,+10])
    #cb.set_label('best reheating scenario')
    #cb.set_label(r'$\left.\ln(R_{\mathrm{reh}})\right|_{\mathrm{best}}$')
    plt.clim(-45,11)
    cb.set_label(cname,fontsize=fscblabel)

    for t in cb.ax.get_yticklabels():
      t.set_fontsize(fslabel)

    #cb.ax.set_yticklabels(['radiation-like'],horizontalalignment='left',fontsize=8,rotation=90)


    #some text: 1.0 0.5 right
    #some text: 1e-3,4.0 left
    if labelname is not None:
      ax0.text(1.0,0.5,labelname, bbox=dict(facecolor='white', alpha=0.8)
               , horizontalalignment='right',verticalalignment='center',zorder=11,fontsize=fslabel)

    plt.savefig(name + '.' +formatname, format=formatname, dpi=150, bbox_inches='tight')


