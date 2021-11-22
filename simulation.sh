#!/bin/bash
# Preliminary phases
printf '[\33[01;32m  phases #1; clean tables  \33[01;37m]\n'
./clean-table.sh #clean old q-table and clock stored locally
# Training
printf '[\33[01;32m  phases #2; training  \33[01;37m]\n'
./gradlew runSwapSourceOnlineMaxBatch
# Evaluation

printf '[\33[01;32m  phases #3; evaluation  \33[01;37m]\n'
printf '[\33[01;32m  phases #3.1; 80 nodes  \33[01;37m]\n'
./gradlew runSwapSourceOnlineTest1Batch
printf '[\33[01;32m  phases #3.1; 360 nodes  \33[01;37m]\n'
./gradlew runSwapSourceOnlineTest2Batch
printf '[\33[01;32m  phases #3.1; 760 nodes  \33[01;37m]\n'
./gradlew runSwapSourceOnlineTest3Batch
# Plots
./plots.sh