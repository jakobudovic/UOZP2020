import gzip
import csv

f = gzip.open("test.csv.gz", "rt")
reader = csv.reader(f, delimiter="\t")
next(reader) #skip legend

fo = open("ultra.txt", "wt")
for l in reader:
    fo.write(l[-3] + "\n")
fo.close()
