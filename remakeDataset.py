import csv
import os

os.rename('newDataset.csv', 'newDataset.csv.bak')
with open('newDataset.csv.bak', 'r') as infile, open('newDataset.csv', 'w') as outfile:
    i = 0
    for line in infile:
        if(i%2 == 0):
            outfile.write(line.replace('\n', ''))
        else:
            outfile.write(line)
        print(line)
        i += 1
os.remove('newDataset.csv.bak')