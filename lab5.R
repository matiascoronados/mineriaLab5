
#install.packages('bnlearn')
require(bnlearn)
####################################################################################
####################################################################################
####################################################################################
######### NUESTRO #########
####################################################################################
####################################################################################
####################################################################################

require(bnlearn)
#install.packages("usethis")
#usethis::use_course("https://goo.gl/x9rdpD")
data(alarm)

bif <- read.bif("/Users/matiascoronado/Desktop/mineriaLab4/ALARM/alarm.bif")
net <- read.net("/Users/matiascoronado/Desktop/mineriaLab4/ALARM/alarm.net")
dsc <- read.dsc("/Users/matiascoronado/Desktop/mineriaLab4/ALARM/alarm.dsc")
rds <- readRDS("/Users/matiascoronado/Desktop/mineriaLab4/ALARM/alarm.rds")

#""""""""""Pre-procesamiento""""""""""
data = alarm[,c(1:37)]

#Lo vamos a entregar.

modelstring = paste0("[HIST|LVF][CVP|LVV][PCWP|LVV][HYP][LVV|HYP:LVF][LVF]",
                     "[STKV|HYP:LVF][ERLO][HRBP|ERLO:HR][HREK|ERCA:HR][ERCA][HRSA|ERCA:HR][ANES]",
                     "[APL][TPR|APL][ECO2|ACO2:VLNG][KINK][MINV|INT:VLNG][FIO2][PVS|FIO2:VALV]",
                     "[SAO2|PVS:SHNT][PAP|PMB][PMB][SHNT|INT:PMB][INT][PRSS|INT:KINK:VTUB][DISC]",
                     "[MVS][VMCH|MVS][VTUB|DISC:VMCH][VLNG|INT:KINK:VTUB][VALV|INT:VLNG]",
                     "[ACO2|VALV][CCHL|ACO2:ANES:SAO2:TPR][HR|CCHL][CO|HR:STKV][BP|CO:TPR]")

dag = model2network(modelstring)
graphviz.plot(dag, layout = "dot")





