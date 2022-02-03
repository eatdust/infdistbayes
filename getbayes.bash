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

#!/bin/bash

if [ $# -eq 0 ]; then
    export NPJOBS=1
else
    export NPJOBS=$1
fi

echo "NPJOBS= $NPJOBS"

#input and output files and directories
export CURRENTDIR=`pwd`
export DISTFILE=$CURRENTDIR/'inifiles/distbayes.ini'
export CHAINDIR=$CURRENTDIR/'chains/'
export PLOTDIR=$CURRENTDIR/'plots/'
export STATDIR=$CURRENTDIR/'stats/'
export DATADIR=$CURRENTDIR/'data/'
export EXECDIR=$CURRENTDIR/'src/'

#compute marginalized distribution for these parameters
export DATAPARAMLIST='lnRreh'

export OUTDIR='./output'
export OUTFILE='bayesdist.log'


#how to get the evidence values in output stat files

#whichZ=-1
export GRABOLDZ="Global Evidence"

#whichZ=0
export GRABNESTZ='Nested Sampling Global Log-Evidence'

#whichZ=1
export GRABINSZ='Nested Importance Sampling Global Log-Evidence'

#whichZ=2
export GRABCHORDZ='log(Z)      '

#use Multinest evidences(0), INS evidences (1) or polychord evidences
#(2) [(-1) for obsoleted multinest versions]
whichZ=0


#how to retrieve the stat files (polychord and multinest output
#differ)
export PREFIX='bayesinf_'

if [ $whichZ == 2 ]; then
    export SUFFIX='.stats'
else
    export SUFFIX='stats.dat'
fi




#command to start infbayes analysis of the samples
export BINDIST='python '$EXECDIR'infbayes.py'
export OPTDIST=' --distfile '$DISTFILE' --chaindir '$CHAINDIR' --outplotdir '$PLOTDIR
export OPTDIST=$OPTDIST' --outstatdir '$STATDIR' --datafor '$DATAPARAMLIST' --datadir '$DATADIR

export EXECDIST=$BINDIST$OPTDIST







function spawndist()
{
    modelname=$1

    if [ $whichZ == 1 ]; then
	GRABINSTATS=$GRABINSZ
    elif [ $whichZ == -1 ]; then
	GRABINSTATS=$GRABOLDZ
    elif [ $whichZ == 2 ]; then
	GRABINSTATS=$GRABCHORDZ
    else
	GRABINSTATS=$GRABNESTZ
    fi

    stats=`grep "$GRABINSTATS" $CHAINDIR/$PREFIX$modelname$SUFFIX \
| sed "s|$GRABINSTATS||g"`

    evidence=`echo $stats | awk '{print $2 +0.0}'`
    error=`echo $stats | awk '{print $4 + 0.0}'`

    inifile=$CHAINDIR/$PREFIX$modelname.ini
    paramfile=$CHAINDIR/$PREFIX$modelname.paramnames
    rangefile=$CHAINDIR/$PREFIX$modelname.ranges

    nlines=`wc -l $inifile | awk '{print $1}'`

#number of aspic parameters, named c1, c2 etc... in the inifile
    naspic=`grep -c 'c' $inifile`


#the number of free parameters depend on the prior space, we reduce by
#one each time the prior is a delta function
    nfreeparams=$[$nlines]
    limitlist=`cat $inifile | awk '{print $1}'`

    for limitname in $limitlist    
    do
	strglow=`cat $inifile | grep -F $limitname | awk '{print $2}'`
	strghigh=`cat $inifile | grep -F $limitname | awk '{print $3}'`	

	if [ "$strglow" == "$strghigh" ]; then
	    nfreeparams=$[$nfreeparams-1]
	    fixedname=`echo $limitname | sed 's|limits\[||g' | sed 's|\]=||g'`
	    fixedparams+=$fixedname' '
	fi
    done


#number of extra parameters
    nextra=$[$nlines-$naspic]


    if [ $modelname == "sr2" ]; then
	naspic=-2
    fi

    if [ $modelname == "sr3" ]; then
	naspic=-3
    fi
    
    #start infbayes analysis
    
    $EXECDIST $modelname  > $OUTDIR/$modelname.log &
    infbayespid=$!
    echo
    echo '-------------------------------------------------------------------'
    echo 'analysing '$modelname'...'
    echo 'running' $EXECDIST $modelname '(pid '$infbayespid')'    
    wait

#get complexity
    complexity=`grep "complexity" $STATDIR/$PREFIX$modelname'_extrastats' \
    	|sed 's|complexity =||g' | awk '{print $1 + 0.0}'`
#derived unconstrained params
    nunconsparams=`echo $nfreeparams $complexity | awk '{print $1 - $2}'`
    
#get best +ln(like)
    bestfit=`grep "Best fit sample" $STATDIR/$PREFIX$modelname'_likestats' \
        | sed 's|Best fit sample -log(Like) =||g' | awk '{print -$1 + 0.0}'`

#dump in output file
    collect="$modelname $nfreeparams $evidence $error $complexity $nunconsparams $bestfit"

    echo $collect \
	| awk '{ printf "%-12s %-12s %-12s %-12s %-12s %-12s %-12s \n",$1,$2,$3,$4,$5,$6,$7 }' \
	>> $OUTFILE
}


#kill calling shell and EXECDIST processes
function cleanup()
{
  echo "**** Killing all $EXECDIST processes! ****"
  kill $!
  killall $EXECDIST
  kill $$
}


# trap keyboard interrupt (control-c)
trap cleanup SIGINT


#################################   Main Action    ##############################

#list a subset of files in chains/
statlist=`ls $CHAINDIR/$PREFIX*$SUFFIX`

#return the list of models
namelist=`echo $statlist | sed 's|'$CHAINDIR/$PREFIX'||g' \
| sed 's|'$SUFFIX'||g'`

#add a header to the output file
echo '#Name #Nparams #Evidence #Error #Complexity #Npars-Comp. #Best' \
    | awk '{ printf "%-12s %-12s %-12s %-12s %-12s %-12s %-12s \n",$1,$2,$3,$4,$5,$6,$7 }' \
	  > $OUTFILE

count=0
for x in $namelist
do
    spawndist $x &
    count=$[$count + 1]
    [[ $((count%NPJOBS)) -eq 0 ]] && wait
done
wait

cat $OUTFILE
echo ''
echo 'getbayes output completed.'

################################################################################



