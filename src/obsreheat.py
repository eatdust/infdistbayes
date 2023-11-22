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
import iobayes as iob
import kldivergence as kl
import reheatfigs as rfig
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
else:
  labelname = None

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
modelname = []
bayesfactor = []
bayesnpar = []
bayesdim = []
bayesfree = []
bayesgain = []
proba = []
norm = 0.0
best = []
mean = []
ffree = []

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

#  print('xmin= xmax= ',xmin,xmax)

  Imean = kl.kullback_leibler(p,q,(xmin,xmax))
  I2mean = kl.kullback_leibler_second_moment(p,q,(xmin,xmax))
  kldiv.append(Imean)
  kldim.append(2.0*(I2mean - Imean*Imean))

  Bfactor = bayesdist['Evidence'][i]-bayesmax
  expBfactor = np.exp(Bfactor)

  modelname.append(bayesdist['Name'][i])
  bayesfactor.append(Bfactor)
  bayesnpar.append(bayesdist['Nparams'][i])
  bayesdim.append(bayesdist['Dimension'][i])
  bayesfree.append(bayesdist['NparamsMinusDim'][i])
  ffree.append(bayesdist['NparamsMinusDim'][i] / bayesdist['Nparams'][i])
  bayesgain.append(bayesdist['DKL'][i])
  proba.append(expBfactor)
  norm += expBfactor

  print('---------------------------------')
  print('For param:  ',param)
  print('KL=       ',kldiv[n])
  print('d=        ',kldim[n])
  print('Best= Mean= ',best[n],mean[n])
  print('+++++++++++++++++++++++++++++++++')
  print('lnB=        ',bayesfactor[n])
  print('KL=         ',bayesgain[n])
  print('n=          ',bayesnpar[n])
  print('d=          ',bayesdim[n])
  print('n-d=        ',bayesfree[n])
  print('---------------------------------')
  n += 1


#Normalize proba and average in model space
nmodel = n
proba = proba/norm

cumul  = 0.0
kldivMean = 0.0
kldivVar = 0.0
kldimMean = 0.0
kldimVar = 0.0

dimMean = 0.0
dimVar = 0.0
nfreeMean = 0.0
nfreeVar = 0.0
gainMean = 0.0
gainVar = 0.0

for n in range(nmodel):
  cumul +=proba[n]
  kldivMean = kldivMean + proba[n]*kldiv[n]
  kldivVar = kldivVar + proba[n]*kldiv[n]*kldiv[n]
  kldimMean = kldimMean + proba[n]*kldim[n]
  kldimVar = kldimVar + proba[n]*kldim[n]*kldim[n]

  dimMean = dimMean + proba[n]*bayesdim[n]
  dimVar = dimVar + proba[n]*bayesdim[n]*bayesdim[n]
  
  nfreeMean = nfreeMean + proba[n]*bayesfree[n]
  nfreeVar = nfreeVar + proba[n]*bayesfree[n]*bayesfree[n]
  
  gainMean = gainMean + proba[n]*bayesgain[n]
  gainVar = gainVar +  proba[n]*bayesgain[n]*bayesgain[n]
  

kldivVar = kldivVar - kldivMean**2
kldimVar = kldimVar - kldimMean**2

dimVar = dimVar - dimMean**2
nfreeVar = nfreeVar - nfreeMean**2
gainVar = gainVar - gainMean**2

print()
print('================================================================')
print('For the selected parameter: ',param)
print('<Dkl>=                      ',kldivMean)
print('sqrt[<Dkl^2> - <Dkl>^2]=    ',np.sqrt(kldivVar))
print('min(Dkl)= max(Dkl)=         ',min(kldiv),max(kldiv))
print('<d>=                        ',kldimMean)
print('sqrt[<d^2> - <d>^2]=        ',np.sqrt(kldimVar))
print('min(d)= max(d)=             ',min(kldim),max(kldim))
print('================================================================')

print()
print('================================================================')
print('Overall statistics')
print('<Dkl>=                      ',gainMean)
print('sqrt[<Dkl^2> - <Dkl>^2]=    ',np.sqrt(gainVar))
print('min(Dkl)= max(Dkl)=         ',min(bayesgain),max(bayesgain))
print('<d>=                        ',dimMean)
print('sqrt[<d^2> - <d>^2]=        ',np.sqrt(dimVar))
print('min(d)= max(d)=             ',min(bayesdim),max(bayesdim))
print('<n-d>=                      ',nfreeMean)
print('sqrt[<(n-d)^2> - <n-d>^2]=  ',np.sqrt(nfreeVar))
print('min(n-d)= max(n-d)=         ',min(bayesfree),max(bayesfree))
print('================================================================')


############################################################################
#output figures
############################################################################

xlabelname = r'Bayes factor $\mathcal{B}/\mathcal{B}_{\mathrm{best}}$'
ylabelname = r'Information gain $D_\mathrm{KL}^{\mathrm{reh}}$ (in bits)'

rfig.create_2d_figure(name=outname+'_Dklreh',lnxmin=-7,lnxmax=0.1,ymin=0.0,ymax=2.8,
                      cname=paramtexname,formatname=formatname,
                      lnxdata=bayesfactor,ydata=kldiv,ydataMean=kldivMean,ydataVar=kldivVar,
                      cdata=mean,sdata=ffree,
                      xlabelname=xlabelname,ylabelname=ylabelname,labelname=labelname,
                      modelname=modelname)

ylabelname = r'Dimensionality $d_{\mathrm{reh}}$'
rfig.create_2d_figure(name=outname+'_dreh',lnxmin=-7.0,lnxmax=0.1,ymin=0.0,ymax=5.0,
                      cname=paramtexname,formatname=formatname,
                      lnxdata=bayesfactor,ydata=kldim,ydataMean=kldimMean,ydataVar=kldimVar,
                      cdata=mean,sdata=None,
                      xlabelname=xlabelname,ylabelname=ylabelname,labelname=labelname,
                      modelname=modelname)


ylabelname = r'Overall information gain $D_\mathrm{KL}$ (in bits)'
rfig.create_2d_figure(name=outname+'_Dkl',lnxmin=-7.0,lnxmax=0.1,ymin=0.0,ymax=10.0,
                      cname=paramtexname,formatname=formatname,
                      lnxdata=bayesfactor,ydata=bayesgain,ydataMean=gainMean,ydataVar=gainVar,
                      cdata=mean,sdata=None,
                      xlabelname=xlabelname,ylabelname=ylabelname,labelname=labelname,
                      modelname=modelname)

ylabelname = r'Bayesian dimensionality $d$'
rfig.create_2d_figure(name=outname+'_d',lnxmin=-7.0,lnxmax=0.1,ymin=0.0,ymax=5.0,
                      cname=paramtexname,formatname=formatname,
                      lnxdata=bayesfactor,ydata=bayesdim,ydataMean=dimMean,ydataVar=dimVar,
                      cdata=mean,sdata=None,
                      xlabelname=xlabelname,ylabelname=ylabelname,labelname=labelname,
                      modelname=modelname)


ylabelname = r'Unconstrained parameters $n-d$'
rfig.create_2d_figure(name=outname+'_nfree',lnxmin=-7,lnxmax=0.1,ymin=-1,ymax=3,
                      cname=paramtexname,formatname=formatname,
                      lnxdata=bayesfactor,ydata=bayesfree,ydataMean=nfreeMean,ydataVar=nfreeVar,
                      cdata=mean,sdata=None,
                      xlabelname=xlabelname,ylabelname=ylabelname,labelname=labelname,
                      modelname=modelname)
