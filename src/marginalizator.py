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
parser.add_argument("--parambound", type=float, nargs=2, help="min/max values for the parameter 'p', as 'pmin pmax'")
parser.add_argument("--paramTeXname", type=str, help="LaTeX name of the parameter (for plot)")
parser.add_argument("--param2Dname", type=iob.split_strings,
                    help="model space 2D marginalization for the couple of parameters 'p1','p2'")
parser.add_argument("--param2Dbound", nargs=4, type=float,
                    help="min/max values for the 2D parameters 'p1','p2' as 'p1min p1max p2min p2max'")
parser.add_argument("--param2DTeXname", type=iob.split_strings,
                    help="LaTeX name of the couple of parameters '$p_1$','$p_2$'")
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

if pargs.parambound is not None:
    parambound = pargs.parambound
else:
    parambound = None
    
if pargs.paramTeXname is not None:
    paramtexname = pargs.paramTeXname
else:
    if (param == 'lnRreh'):
        paramtexname = r'$\ln R_{\mathrm{reh}}$'
    else:
        paramtexname = r'$p$'


if pargs.param2Dname is not None:
    param2D = pargs.param2Dname
else:
    param2D = None 

if pargs.param2Dbound is not None:
    param2Dbound = pargs.param2Dbound
else:
    param2Dbound = None 
    
if pargs.param2DTeXname is not None:
    param2Dtexname = pargs.param2DTeXname
else:
    param2Dtexname = None
    

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
priors2D = []
posteriors2D = []

#read evidences
bayesdist = iob.load_bayesdist(pargs.bayesdistname)

bayesmax = np.amax(bayesdist['Evidence'])

  
#read and process margestats, likestats, prior and posterior files
for i in range(bayesdist.shape[0]):

    Bfactor = bayesdist['Evidence'][i]-bayesmax
    expBfactor = np.exp(Bfactor)

    modelname.append(bayesdist['Name'][i])
    bayesfactor.append(Bfactor)
    proba.append(expBfactor)
    norm += expBfactor

    
    print()
    print('Model: ',i,bayesdist['Name'][i])    

    
    postname = datadir + rootname + bayesdist['Name'][i] + '_p_' + param + '.dat'
    priorname = nulldir + rootname + bayesdist['Name'][i] + '_p_' + param + '.dat'
    exist = os.path.isfile(postname) and os.path.isfile(priorname)
    if not exist:
        print('Some data file are missing for model: ',bayesdist['Name'][i])
        print(postname+': ',os.path.isfile(postname))
        print(priorname+': ',os.path.isfile(priorname))
        raise Exception('density not found!')
          
    xprior, prior, unprior = iob.load_and_normalize_probability(priorname)
    xpost, post, unpost = iob.load_and_normalize_probability(postname)

    priors.append([xprior,prior])
    posteriors.append([xpost,post])

    
    if param2D is not None:
        file2Dparam = datadir+rootname+ bayesdist['Name'][i] + '_pp_' + param2D[0]+'_'+ param2D[1] + '.dat'
        post2Dname = datadir+rootname+ bayesdist['Name'][i] + '_2D_' + param2D[0]+'_'+ param2D[1] + '.dat'
        prior2Dname = nulldir+rootname+ bayesdist['Name'][i] + '_2D_' + param2D[0]+'_'+ param2D[1] + '.dat'
        exist = os.path.isfile(file2Dparam) and os.path.isfile(post2Dname) and os.path.isfile(prior2Dname)
      
        if not exist:

            print('Some 2D data file are missing for model: ',bayesdist['Name'][i])
            print(file2Dparam+': ',os.path.isfile(file2Dparam))
            print(post2Dname+': ',os.path.isfile(post2Dname))
            print(prior2Dname+': ',os.path.isfile(prior2Dname))
            raise Exception('density2D not found!')

        xprior2D, yprior2D, prior2D, unprior2D= iob.load_and_normalize_probability_2d(prior2Dname,file2Dparam)
        xpost2D, ypost2D, post2D, unpost2D= iob.load_and_normalize_probability_2d(post2Dname,file2Dparam)
        
        priors2D.append([xprior2D,yprior2D,prior2D])
        posteriors2D.append([xpost2D,ypost2D,post2D])
        


  
    print('---------------------------------')
    print('lnB =        ',bayesfactor[n])
    print('param name  :',param)
    print('posterior   :',postname)
    print('prior       :',priorname)
    if param2D is not None:
        print('param2D     :',file2Dparam)
        print('posterior 2D:',post2Dname)
        print('prior 2D    :',prior2Dname)
    print('---------------------------------')
    n += 1

  
#Normalize proba and average in model space
nmodel = n
proba = proba/norm

if (param == 'lnRrad'):
    xminprior = -46
    xmaxprior = 15
    xminpost = -46
    xmaxpost = 15
    paramtexname = r'$\ln R_{\mathrm{rad}}$'

if (param == 'lnRreh'):
    xminprior = -46
    xmaxprior = 15
    xminpost = -46
    xmaxpost = 15
    
if (param == 'alpha'):
    xminprior = -0.01
    xmaxprior = 0.003
    xminpost = -0.0025
    xmaxpost = 0.0005
    paramtexname = r'$\alpha_\mathrm{S}$'
    
if (param == 'wreh'):
    xminprior = -0.33
    xmaxprior = 1.0
    xminpost = -0.33
    xmaxpost = 1.0
    paramtexname = r'$\bar{w}_\mathrm{reh}$'
    
if (param == 'lnRhoReh'):
    xminprior = -187
    xmaxprior = 0.0
    xminpost = -187
    xmaxpost = 0.0
    paramtexname = r'$\ln \rho_\mathrm{reh}$'

if (parambound is not None):
    xminprior = parambound[0]
    xmaxprior = parambound[1]
    xminpost = parambound[0]
    xmaxpost = parambound[1]
    
    
units = np.ones(nmodel)/nmodel

titlename = 'Normalized prior distribution (model space)'
ylabelname = r'$\pi\left('+paramtexname.replace('$', '')+r'\right)$'
pfig.create_1d_figure(name=param+'_priors',distrib=priors,weight=units,xmin=xminprior,xmax=xmaxprior,
                      xlabelname=paramtexname,ylabelname=ylabelname,titlename=titlename,formatname=formatname,save=True)


ylabelname = r'$P\left('+paramtexname.replace('$', '')+r'|\mathcal{D}\right)$'
titlename = 'Normalized posterior distribution (model space)'        
pfig.create_1d_figure(name=param+'_posteriors',distrib=posteriors,weight=proba,xmin=xminpost,xmax=xmaxpost,
                      xlabelname=paramtexname,ylabelname=ylabelname,titlename=titlename,formatname=formatname,save=True)


if param2D is not None:

    for i in range(len(param2D)):
        if param2D[i] == 'wreh':
            if param2Dtexname is None:
                param2Dtexname = [None,None]
            param2Dtexname[i] = r'$\bar{w}_{\mathrm{reh}}$'

        if param2D[i] == 'lnRhoReh':
            if param2Dtexname is None:
                param2Dtexname = [None,None]
            param2Dtexname[i] = r'$\ln \rho_\mathrm{reh}$'

                
    if param2Dtexname is not None:
        xlabelname = r''+param2Dtexname[0]
        ylabelname = r''+param2Dtexname[1]
        clabelpriorname = r'$\pi\left('+param2Dtexname[0].replace('$', '')+','+param2Dtexname[1].replace('$', '')+r'\right)$'
        clabelpostname = r'$P\left('+param2Dtexname[0].replace('$', '')+','+param2Dtexname[1].replace('$', '')+r'|\mathcal{D}\right)$'
    else:
        xlabelname = None
        ylabelname = None
        clabelpriorname = None
        clabelpostname = None

    if param2Dbound is not None:
        xmin = param2Dbound[0]
        xmax = param2Dbound[1]
        ymin = param2Dbound[2]
        ymax = param2Dbound[3]
    else:
        xmin = None
        xmax = None
        ymin = None
        ymax = None
        
    titlename = 'Normalized prior distribution (model space)'
    pfig.create_2d_figure(name='priors_2D_'+param2D[0]+'_'+param2D[1],distrib2D=priors2D,weight=units,
                          xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,
                          xlabelname=xlabelname,ylabelname=ylabelname,clabelname=clabelpriorname,
                          titlename=titlename,formatname=formatname)

    
    titlename = 'Normalized posterior distribution (model space)'
    pfig.create_2d_figure(name='posteriors_2D_'+param2D[0]+'_'+param2D[1],distrib2D=posteriors2D,weight=proba,
                          xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,
                          xlabelname=xlabelname,ylabelname=ylabelname,clabelname=clabelpostname,
                          titlename=titlename,formatname=formatname)
