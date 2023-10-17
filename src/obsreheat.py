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

#!/usr/bin/python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import iobayes as iob
import kldivergence as kl
import argparse
import os


parser = argparse.ArgumentParser()

parser.add_argument("bayesdistname", help="Bayesian evidence file name")

parser.add_argument("--rootname", type=str, help="root name of all read files (bayesinf_)")
parser.add_argument("--statdir", type=str, help="where to read the marge and like statistics")
parser.add_argument("--paramname", type=str, help="information gain for the named parameter")
parser.add_argument("--paramTeXname", type=str, help="LaTeX name of the parameter (for plot)")
parser.add_argument("--nulldir", type=str, help="where to read the marginalized priors")
parser.add_argument("--datadir", type=str, help="where to read the marginalized probabilities")
parser.add_argument("--plottype", type=str, help="image extension (eps, png, ...)")
parser.add_argument("--labelname", type=str, help="label to show of the final figure")
parser.add_argument("--outname", type=str, help="name of the output file")

pargs = parser.parse_args()


if pargs.rootname is not None:
  rootname = pargs.rootname
else:
  rootname = 'bayesinf_'


if pargs.statdir is not None:
  statdir = pargs.statdir
else:
  statdir = "stats/"


if pargs.paramname is not None:
  param = pargs.paramname
else:
  param = 'lnRreh'

if pargs.paramTeXname is not None:
  paramtexname = pargs.paramTeXname
else:
  paramtexname = r'$\left\langle \ln R_{\mathrm{reh}} \right\rangle$'


if pargs.nulldir is not None:
  nulldir = pargs.nulldir
else:
  nulldir = 'null/'
  
if pargs.datadir is not None:
  datadir = pargs.datadir
else:
  datadir = 'data/'

if pargs.plottype is not None:
  formatname = pargs.plottype
else:
  formatname = 'png'

if pargs.labelname is not None:
  labelname = pargs.labelname
  addlabelname = True
else:
  addlabelname = False

if pargs.outname is not None:
  outname = pargs.outname
else:
  outname = 'obsreheating'
  

############################################################################
#reading data and computing information gain
############################################################################
  

#outname='plc_2'
#outname="corem5_mhi_D"
#outname='litebirdhi'


#labelname='CORE-M5 MHI Delensed'
#labelname='Planck 2015 + BICEP2/KECK'
#labelname='LITEBIRD SI + Planck 2013'


n = 0
kldiv = []
kldim = []
bayesfactor = []
proba = []
norm = 0.0
best = []
mean = []

#read evidences
bayesdist = iob.load_bayesdist( pargs.bayesdistname)

bayesmax = np.amax(bayesdist['Evidence'])

  
#read and process margestats, likestats, prior and posterior files
for i in range(bayesdist.shape[0]):
  margename = statdir + rootname + bayesdist['Name'][i] + '_margestats'
  likename = statdir + rootname + bayesdist['Name'][i] + '_likestats'
  postname = datadir + rootname + bayesdist['Name'][i] + '_p_' + param + '.dat'
  priorname = nulldir + rootname + bayesdist['Name'][i] + '_p_' + param + '.dat'

  exist = os.path.isfile(postname) and os.path.isfile(priorname)
  exist = exist and os.path.isfile(margename) and os.path.isfile(likename)

  print()
  print('Model: ',i,bayesdist['Name'][i])

  if not exist:
    print('Some data file are missing for model: ',bayesdist['Name'][i])
    print(postname+': ',os.path.isfile(postname))
    print(priorname+': ',os.path.isfile(priorname))
    print(margename+': ',os.path.isfile(margename))
    print(likename+': ',os.path.isfile(likename))
    continue

  
  margestats = iob.load_margestats(margename)
  
  for j in range(margestats.shape[0]):
    if margestats['parameter'][j] == param:
      mean.append(margestats['mean'][j])
        

  likestats = iob.load_likestats(likename)
    
  for j in range(likestats.shape[0]):
    if likestats['parameter'][j] == param:
      best.append(likestats['bestfit'][j])

        
  xprior, prior = iob.load_and_normalize_probability(priorname)
  xpost, post = iob.load_and_normalize_probability(postname)

  q = kl.interp_probability_density(xprior,prior)
  p = kl.interp_probability_density(xpost,post)
  xmin = max(min(xprior),min(xpost))
  xmax = min(max(xprior),max(xpost))

  print('xmin= xmax= ',xmin,xmax)

  Imean = kl.kullback_leibler(p,q,(xmin,xmax))
  I2mean = kl.kullback_leibler_second_moment(p,q,(xmin,xmax))
  kldiv.append(Imean)
  kldim.append(2.0*(I2mean - Imean*Imean))

  Bfactor = bayesdist['Evidence'][i]-bayesmax
  expBfactor = np.exp(Bfactor)

  bayesfactor.append(Bfactor)
  proba.append(expBfactor)
  norm += expBfactor

  print('KL=  ',kldiv[n])
  print('d=   ',kldim[n])
  print('lnB= ',bayesfactor[n])
  print('Param is: ',param)
  print('Best= Mean= ',best[n],mean[n])
       
  n += 1


#Normalize proba and average in model space
nmodel = n
proba = proba/norm

kldivMean = 0.0
kldivVar = 0.0
kldimMean = 0.0
kldimVar = 0.0
cumul  = 0.0

for n in range(nmodel):
  cumul +=proba[n]
  kldivMean = kldivMean + proba[n]*kldiv[n]
  kldivVar = kldivVar + proba[n]*kldiv[n]*kldiv[n]
  kldimMean = kldimMean + proba[n]*kldim[n]
  kldimVar = kldimVar + proba[n]*kldim[n]*kldim[n]

kldivVar = kldivVar - kldivMean**2
kldimVar = kldimVar - kldimMean**2

print()
print('================================================================')
print('<Dkl>= ',kldivMean)
print('sqrt[<Dkl^2> - <Dkl>^2]= ',np.sqrt(kldivVar))
print('min(Dkl)= max(Dkl)= ',min(kldiv),max(kldiv))
print('<d>= ',kldimMean)
print('sqrt[<d^2> - <d>^2]= ',np.sqrt(kldimVar))
print('min(d)= max(d)= ',min(kldim),max(kldim))
print('================================================================')

############################################################################
#output figures
############################################################################

mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['mathtext.fontset'] = 'cm'

fig, ax0 = plt.subplots()
plt.axis([0,1,0,1])

lnxmin = -7.0
lnxmax = 0.1

xmin = np.exp(lnxmin)
xmax = np.exp(lnxmax)
ymin = 0.0
ymax = 2.8
#ymax = 5

fsjeffrey = 8
fslabel = 12
fscblabel=14

#ax0.set_title('Observing the inflationary reheating',fontsize=8)

ax0.set_xlabel(r'Bayes factor $\mathcal{B}/\mathcal{B}_{\mathrm{best}}$',fontsize=fslabel)
ax0.set_xscale('log')
ax0.set_xlim(xmin, xmax)
ax0.set_ylabel(r'Information gain $D_\mathrm{KL}$ (in bits)',fontsize=fslabel)
ax0.set_ylim(ymin, ymax)

#color scale
c = plt.scatter(np.exp(bayesfactor),kldiv,c=mean,s=100,
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



ymean = kldivMean
ystddev= np.sqrt(kldivVar)
yone = ymean - 0.5*ystddev
ytwo = ymean + 0.5*ystddev

ax0.fill_between(x=[xmin,xmax],y1=[yone,yone],y2=[ytwo,ytwo],facecolor='yellow', alpha=0.8)

if outname == 'plc_2':
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
cb.set_label(paramtexname,fontsize=fscblabel)

for t in cb.ax.get_yticklabels():
  t.set_fontsize(fslabel)

#cb.ax.set_yticklabels(['radiation-like'],horizontalalignment='left',fontsize=8,rotation=90)


#some text: 1.0 0.5 right
#some text: 1e-3,4.0 left
if addlabelname:
  ax0.text(1.0,0.5,labelname, bbox=dict(facecolor='white', alpha=0.8)
           , horizontalalignment='right',verticalalignment='center',zorder=11,fontsize=fslabel)

  
#######################################

formatname='png'
plt.savefig(outname + '.' +formatname, format=formatname, dpi=150, bbox_inches='tight')
