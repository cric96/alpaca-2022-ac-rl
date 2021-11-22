#!/bin/bash
## Training error evolution
printf '[\33[01;32m  phases #4; plots  \33[01;37m]\n'
printf '[\33[01;32m  phases #4.1; training error evolution  \33[01;37m]\n'
amm src/main/plot/average-value.sc --skip 0 --experimentName swapSourceOnlineMax 4 5
## Evaluation plots
printf '[\33[01;32m  phases #4.2; evaluation average error \33[01;37m]\n'
printf '[\33[01;32m  phases #4.2.1; 80 nodes \33[01;37m]\n'
amm src/main/plot/bulk-plots.sc plotOnly --output average-error-80 --baseExperiment swapSourceOnlineTest1 --configuration src/main/plot/swapSource-80.yml
printf '[\33[01;32m  phases #4.2.1; 360 nodes \33[01;37m]\n'
amm src/main/plot/bulk-plots.sc plotOnly --output average-error-360 --baseExperiment swapSourceOnlineTest2 --configuration src/main/plot/swapSource-360.yml
printf '[\33[01;32m  phases #4.2.1; 760 nodes \33[01;37m]\n'
amm src/main/plot/bulk-plots.sc plotOnly --output average-error-760 --baseExperiment swapSourceOnlineTest3 --configuration src/main/plot/swapSource-760.yml