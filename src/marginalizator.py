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
import postfigs as pfig
import argparse
import os


parser = argparse.ArgumentParser()

parser.add_argument("bayesdistname", help="Bayesian evidence file name")

parser.add_argument("--rootname", type=str, help="root name of all read files (bayesinf_)")
parser.add_argument("--paramname", type=str, help="model space marginalization for the named parameter 'p'")
parser.add_argument("--paramTeXname", type=str, help="LaTeX name of the parameter (for plot)")
parser.add_argument("--param2Dnames", type=str, help="model space 2D marginalization for the couple of parameters 'p1','p2'")
parser.add_argument("--param2DTeXnames", type=str, help="LaTeX name of the couple of parameters '$p_1$','$p_2$'")
parser.add_argument("--nulldir", type=str, help="where to read the marginalized priors")
parser.add_argument("--datadir", type=str, help="where to read the marginalized probabilities")
parser.add_argument("--plottype", type=str, help="image extension (eps, png, ...)")

pargs = parser.parse_args()


if pargs.rootname is not None:
    rootname = pargs.rootname
else:
    rootname = 'bayesinf_'


if pargs.paramname is not None:
    param = pargs.paramname
else:
    param = 'lnRreh'

if pargs.paramTeXname is not None:
    paramtexname = pargs.paramTeXname
else:
    if (pargs.paramname is None) or (param == 'lnRreh'):
        paramtexname = r'$\ln R_{\mathrm{reh}}$'
    else:
        paramtexname = r'$p$'
        

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


############################################################################
#loading evidences, prior and posteriors
############################################################################


n = 0
modelname = []
bayesfactor = []
proba = []
norm = 0.0
priors = []
posteriors = []

#read evidences
bayesdist = iob.load_bayesdist(pargs.bayesdistname)

bayesmax = np.amax(bayesdist['Evidence'])

  
#read and process margestats, likestats, prior and posterior files
for i in range(bayesdist.shape[0]):
  postname = datadir + rootname + bayesdist['Name'][i] + '_p_' + param + '.dat'
  priorname = nulldir + rootname + bayesdist['Name'][i] + '_p_' + param + '.dat'

  exist = os.path.isfile(postname) and os.path.isfile(priorname)

  print()
  print('Model: ',i,bayesdist['Name'][i])

  if not exist:
    print('Some data file are missing for model: ',bayesdist['Name'][i])
    print(postname+': ',os.path.isfile(postname))
    print(priorname+': ',os.path.isfile(priorname))
    continue
          
  xprior, prior, unprior = iob.load_and_normalize_probability(priorname)
  xpost, post, unpost = iob.load_and_normalize_probability(postname)

  Bfactor = bayesdist['Evidence'][i]-bayesmax
  expBfactor = np.exp(Bfactor)

  modelname.append(bayesdist['Name'][i])
  bayesfactor.append(Bfactor)
  proba.append(expBfactor)
  norm += expBfactor

  priors.append([xprior,prior])
  posteriors.append([xpost,post])
  
  print('---------------------------------')
  print('lnB =        ',bayesfactor[n])
  print('param name:  ',param)
  print('posterior :  ',postname)
  print('prior     :  ',postname)        
  print('---------------------------------')
  n += 1


#Normalize proba and average in model space
nmodel = n
proba = proba/norm

if ((param == 'lnRrad') or (param == 'lnRreh')):
    xminprior = -46
    xmaxprior = 15
    xminpost = -46
    xmaxpost = 15

if (param == 'alpha'):
    xminprior = -0.01
    xmaxprior = 0.003
    xminpost = -0.0025
    xmaxpost = 0.0005

if (param == 'wreh'):
    xminprior = -0.33
    xmaxprior = 1.0
    xminpost = -0.33
    xmaxpost = 1.0

if (param == 'lnRhoReh'):
    xminprior = -187
    xmaxprior = 0.0
    xminpost = -187
    xmaxpost = 0.0
    
    
units = np.ones(nmodel)/nmodel

titlename = 'Normalized prior distribution (model space)'
ylabelname = r'$\pi\left('+paramtexname.replace('$', '')+r'\right)$'
pfig.create_1d_figure(name=param+'_priors',distrib=priors,weight=units,xmin=xminprior,xmax=xmaxprior,
                      xlabelname=paramtexname,ylabelname=ylabelname,titlename=titlename,formatname=formatname,save=True)


ylabelname = r'$P\left('+paramtexname.replace('$', '')+r'|\mathcal{D}\right)$'
titlename = 'Normalized posterior distribution (model space)'        
pfig.create_1d_figure(name=param+'_posteriors',distrib=posteriors,weight=proba,xmin=xminpost,xmax=xmaxpost,
                      xlabelname=paramtexname,ylabelname=ylabelname,titlename=titlename,formatname=formatname,save=True)
