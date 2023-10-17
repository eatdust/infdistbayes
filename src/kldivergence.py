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
from scipy import interpolate, integrate

def interp_probability_density(x,y):
    finterp = interpolate.interp1d(x,y,kind='linear',bounds_error=False,
                                fill_value=0.0)
    return finterp


def kullback_leibler(p,q,bounds,inbit=True):
    tol = 1e-3
    args=(p,q,inbit)
    KL,err = integrate.quad(kl_integrand,bounds[0],bounds[1],args,limit=1000,epsabs=tol,epsrel=tol)
    if (err > tol):
        print('kullback_leibler KL= error=: ',KL,err)
    return KL        


def kl_integrand(x,p,q,inbit):
    if p(x) <= 0 or q(x) <= 0:
#        print('x= p(x)= q(x)= ',x,p(x))
        return 0.0
    else:
        if inbit:
            return p(x)*np.log2(p(x)/q(x))
        else:
            return p(x)*np.log(p(x)/q(x))


def kullback_leibler_second_moment(p,q,bounds,inbit=True):
    tol = 1e-3
    args=(p,q,inbit)
    KL2,err = integrate.quad(kl_second_moment_integrand,bounds[0],bounds[1],args,limit=1000,epsabs=tol,epsrel=tol)
    if (err > tol):
        print('kullback_leibler_second_moment KL^2= error=: ',KL2,err)
    return KL2


def kl_second_moment_integrand(x,p,q,inbit):
    if p(x) <= 0 or q(x) <= 0:
#        print('x= p(x)= q(x)= ',x,p(x))
        return 0.0
    else:
        if inbit:
            return p(x)*( np.log2(p(x)/q(x)) )**2
        else:
            return p(x)*( np.log(p(x)/q(x)) )**2
