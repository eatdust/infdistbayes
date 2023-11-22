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
import matplotlib.path as mpp
import matplotlib.patches as mpt
import matplotlib.transforms as mtr


class ModelBoxStyle:
    """A model box."""

    def __init__(self, pad=0.3, radius=6, angle=None):

        self.pad = pad
        self.angle = angle
        self.radius = radius
        
        super().__init__()

    def __call__(self, x0, y0, width, height, mutation_size):

        pad = mutation_size * self.pad
        width, height = width + 2 * pad, height + 2 * pad
        x0, y0 = x0 - pad, y0 - pad

        radius = 0.25 * mutation_size * self.radius
        
        modelbox = mpp.Path.circle(center=(x0 + width / 2, y0 + height / 2),radius=radius)


        if self.angle is not None:
            ww = self.pad * radius
            rw = radius + ww
            origin = 90.0 
            angle = 90.0 + self.angle

            wedge = mpt.Wedge(center=(x0 + width / 2, y0 + height / 2), r=rw,
                              theta1=origin, theta2=angle, width=ww).get_path()

                
            modelbox = mpp.Path.make_compound_path(modelbox,wedge)
        

        return modelbox
            


def set_figure_params(dpi=None):

    if dpi is not None:
        mpl.rcParams['figure.dpi'] = dpi
        
    #register our new boxtyle
    mpl.patches.BoxStyle._style_list["aspicmodel"] = ModelBoxStyle
    

def create_2d_figure(name,lnxmin,lnxmax,ymin,ymax,cname,formatname,
                     lnxdata,ydata,ydataMean,ydataVar,cdata,sdata=None,
                     xlabelname=None,ylabelname=None,labelname=None,modelname=None):

    set_figure_params(dpi=200)

    fig, ax0 = plt.subplots()
    plt.axis([0,1,0,1])

    xmin = np.exp(lnxmin)
    xmax = np.exp(lnxmax)
    xdata = np.exp(lnxdata)

    fsmodel = 4
    fsjeffrey = 8
    fslabel = 12
    fscblabel = 14
    
    ax0.set_xlabel(xlabelname,fontsize=fslabel)
    ax0.set_xscale('log')
    ax0.set_xlim(xmin, xmax)
    ax0.set_ylabel(ylabelname,fontsize=fslabel)
    ax0.set_ylim(ymin, ymax)
    
    if modelname is not None:
        
        norm = mpl.colors.Normalize(vmin=min(cdata),vmax=max(cdata))
        cmap = mpl.cm.jet
        c = mpl.cm.ScalarMappable(norm=norm,cmap=cmap)

        boxstyle = "aspicmodel,pad=0.3,radius=5.5"
        
        for i,aname in enumerate(modelname):
            xlab = xdata[i]
            ylab = ydata[i]
            lab = "{:^4}".format(aname[:4])
            col = cmap(norm(cdata[i]))
            textcol = (1-col[0],1-col[1],1-col[2],col[3])

            if sdata is not None:
                strvalue = "{:}".format(max(0.0,360.0*sdata[i]))
                aboxstyle = boxstyle + ",angle=" + strvalue
            else:
                aboxstyle = boxstyle
            
            ax0.annotate(lab, xy=(xlab, ylab), xytext=(0,0),textcoords='offset points',
                         va="center", ha="center",fontsize=fsmodel,color=textcol,zorder=10,
                         bbox=dict(boxstyle=aboxstyle,linewidth=0.2,
                                   edgecolor='black',facecolor=col,alpha=1.0))

            
    else:
        
        c = plt.scatter(xdata,ydata,s=100,c=cdata,
                    linewidths=0.5,edgecolors='black',cmap='jet',zorder=10)
            

    #legends and labels (from Totor's colormap)
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


    extrafrac = 0.05
    ytext=ymax + extrafrac*(ymax-ymin)
    xinc = np.exp(Einc+0.5*(lnxmax-Einc))
    xweak = np.exp(Eweak+0.5*(Einc-Eweak))
    xmod = np.exp(Emod + 0.5*(Eweak-Emod))
    xstg = np.exp(lnxmin + 0.5*(Emod-lnxmin))

    if (xmin < xinc < xmax):
        ax0.text(xinc,ytext, 'favored', color='black',fontsize=fsjeffrey
                 ,horizontalalignment='center',verticalalignment='center')

    if (xmin < xweak < xmax):
        ax0.text(xweak,ytext, 'weakly disfavored', color='black',fontsize=fsjeffrey
                 ,horizontalalignment='center',verticalalignment='center')

    if (xmin < xmod < xmax):
        ax0.text(xmod,ytext, 'moderately disfavored', color='black',fontsize=fsjeffrey
                 ,horizontalalignment='center',verticalalignment='center')

    if (xmin < xstg < xmax):
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
    c.set_clim(-45,11)
    cb = fig.colorbar(c,orientation='vertical',pad=0.01,aspect=30,fraction=0.05
                          ,ticks=[-10,-20,-30,-40,0,+10],ax=ax0)
    cb.set_label(cname,fontsize=fscblabel)    
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fslabel)

    if labelname is not None:
        ax0.text(1.0,0.5,labelname, bbox=dict(facecolor='white', alpha=0.8)
                 , horizontalalignment='right',verticalalignment='center',zorder=11,fontsize=fslabel)

#setting bbox for postscript messes the colorbar        
    if (formatname == 'eps' or formatname == 'ps'):
        plt.savefig(name + '.' +formatname, format=formatname)
    else:
        plt.savefig(name + '.' +formatname, format=formatname, bbox_inches='tight')


