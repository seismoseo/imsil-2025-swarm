#!/bin/sh

model="kim2011"
region="Imsil"

hyp1.40 <<EOF

REP T T
CON 50
MIN 4
ZTR 10 F
DIS 4 50 1 3
RMS 4 .12 2 4

* OUTPUT FORMAT
ERF T					/Send error messages to terminal
TOP F					/No page ejects
* LST 0
LST 2 0 1				/station list or models in printfile
KPR 3				    /Medium print output each event
H71 4 1 3		        /Use hypo71 summary format for this test

* STATION DATA
TYP Read in stations:
STA 'STA/${region}_hyp.sta'    			 /Read ASCII file of all stations.

* CRUSTAL MODEL
TYP Read in crustal model(s):
CRH 1 '${model}/${model}_p.crh'
CRH 2 '${model}/${model}_s.crh'
SAL 1 2
* PSM 1 1.73 0

* SET DEFAULT PHASE FILE BUT GIVE USER A CHANCE TO CHANGE IT
TYP Hit enter or carriage-return to keep the test filenames:
TYP (Also tests the phase file to use its format)
PHS 'PHS/${region}.phs'
FIL                                      /Examine phase file and set its format

* SET DEFAULT PRINT FILE BUT GIVE USER A CHANCE TO CHANGE IT
PRT '${model}/${region}.prt'

* SET DEFAULT SUMMARY FILE BUT GIVE USER A CHANCE TO CHANGE IT
SUM '${model}/${region}.sum'

* SET DEFAULT ARCHIVE FILE BUT GIVE USER A CHANCE TO CHANGE IT
ARC '${model}/${region}.arc'

TYP 'Now type LOC to locate the earthquake'
LOC

STO

EOF
