import gzip
import csv
import lpputils

f = gzip.open("test.csv.gz", "rt")
reader = csv.reader(f, delimiter="\t")
next(reader) #skip legend

fo = open("polurniki.txt", "wt")
for l in reader:
    fo.write(lpputils.tsadd(l[-3], 30*60) + "\n")
fo.close()
