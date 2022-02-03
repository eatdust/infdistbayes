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
import getdist.plots as gp
import getdist.mcsamples as gm
import iobayes as iob
import argparse
#import aspicdist as ad


parser = argparse.ArgumentParser()

parser.add_argument("name", help="model name")

parser.add_argument("--distfile", type=str, help="full path to the dist configuration file")
parser.add_argument("--chaindir", type=str, help="path to the chains/ directory")
parser.add_argument("--outplotdir", type=str, help="where to put the plots")
parser.add_argument("--outstatdir", type=str, help="where to put the marge and like statistics")
parser.add_argument("--plottype", type=str, help="image extension (eps, png, ...)")
parser.add_argument("--datafor", type=str, nargs='+', help="output marginalized probality for the named parameters")
parser.add_argument("--datadir", type=str, help="where to dump the marginalized probabilities")




pargs = parser.parse_args()

rootname = 'bayesinf_' + pargs.name

if pargs.distfile is not None:
    distname = pargs.distfile
else:
    distname = "inifiles/distbayes.ini"

if pargs.chaindir is not None:
    chaindirname = pargs.chaindir
else:
    chaindirname = "chains/"

if pargs.outplotdir is not None:
    outplotdir = pargs.outplotdir
else:
    outplotdir = "plots/"

if pargs.outstatdir is not None:
    outstatdir = pargs.outstatdir
else:
    outstatdir = "stats/"

if pargs.plottype is not None:
    formatname = pargs.plottype
else:
    formatname = 'png'

if pargs.datadir is not None:
    datadir = pargs.datadir
else:
    datadir = "data/"

    
sampler = 'nested'
#sampler = 'polychord'    

mc =  gm.loadMCSamples(file_root=chaindirname+rootname, ini=distname, no_cache=True)
print('Having ignored rows: ', mc.ignore_rows)

mc.sampler = sampler
print('Sampler set to: ',mc.sampler)

#multinest output -2 ln(like) as opposed to mcmc which output
#-ln(like). Getdist expects -ln(like) so we rescale the loglike, but
#not the weights (this is different than cooling the chain)
if mc.sampler == 'nested' or mc.sampler == 'polychord':
    mc.loglikes = mc.loglikes/2.0
    mc.updateBaseStatistics()

    
#### stats

print(mc.getNumSampleSummaryText())
print('Num of non-derived parameters=',mc.paramNames.numNonDerived())

#contourname = outdir + rootname
#ad.contour_curves_2d(contourname,mc,'sr2','sr1')

print('Getting margestats...')
margestats = mc.getMargeStats()
margename = outstatdir+rootname+'_margestats'
print(margestats,file=open(margename, 'w'))
print('saved as: ',margename)

print('Getting likestats...')
likestats = mc.getLikeStats()
likename = outstatdir+rootname+'_likestats'
print('saved as: ',likename)
print(likestats,file=open(likename, 'w'))

#the maxlike is hardly well determined, we can define a complexity
#from the log(meanlike) that traces the highest values of the
#likelihood only
deltamean = 2.0*(likestats.meanLogLike - likestats.logMeanLike)
extraname = outstatdir+rootname+'_extrastats'
print('complexity =',likestats.complexity,file=open(extraname,'w'))
print('deltamean =',deltamean,file=open(extraname,'a'))
print('saved as: ',extraname)


#### marginalized 1D distribution

if pargs.datafor is not None:
    print('Getting marginalized 1D distribution...')

    for param in pargs.datafor:
        density = mc.get1DDensityGridData(param,meanlike=False)
        if density is not None:
            filename = datadir + rootname + '_p_' + param + '.dat'
            iob.save_probability_1d(filename,density.x,density.P)
            print('saved as: ',filename)
        else:
            print('parameter is: ',param)
            raise Exception('density not found!')
        



#### plots

print('Getting paramnames...')
params1D = mc.getParamNames().getRunningNames() + mc.getParamNames().getDerivedNames()
print('params1D= ',params1D)

print('Getting correlation matrix...')
mc.getCorrelationMatrix()
params2D = mc.getCorrelatedVariable2DPlots()
params2D = params2D + [['lnA','lnM'],['logr','ns'],['lnRhoEnd','lnRreh'],
            ['lnM','bfold'],['lnRhoEnd','bfold'],
            ['lnRreh','lnRrad'],['lnRreh','ns']]

if ('eps2' in params1D) and ('eps3' in params1D):
    params2D = params2D + [['eps2','logeps'],['eps3','logeps'],['eps2','eps3']]
    

print('params2D= ',params2D)




g1=gp.get_subplot_plotter()
g1.settings.scaling=True
g1.settings.set_with_subplot_size(4.0000)
g1.settings.legend_frac_subplot_margin = 0.1
g1.settings.alpha_filled_add = 0.6
g1.settings.solid_contour_palefactor = 0.6
g1.settings.solid_colors = [('#8CD3F5', '#006FED'), ('#F7BAA6', '#E03424'), ('#D1D1D1', '#A1A1A1'), 'g', 'cadetblue', 'indianred']
#g.settings.solid_colors = ['indianred','darkorange','forestgreen','SteelBlue']

g2=g1

g1.plots_1d(mc,params=params1D, shaded=False, filled=True)
g1.export(outplotdir+rootname+'.'+formatname)

g2.plots_2d(mc,param_pairs=params2D,nx=4,shaded=False, filled=True)
g2.export(outplotdir+rootname+'_2D.'+formatname)
