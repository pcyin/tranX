#!/bin/bash

echo "download dataset"
data_file="tranx.0.2.0.zip"
wget -c http://www.cs.cmu.edu/~pengchey/${data_file}
unzip ${data_file}
rm ${data_file}

for dataset in django geo atis jobs wikisql conala;
do
	mkdir -p saved_models/${dataset}
	mkdir -p logs/${dataset}
	mkdir -p decodes/${dataset}
done

echo "Done!"
