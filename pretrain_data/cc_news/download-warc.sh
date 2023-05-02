#!/bin/bash

# Dump is yyyy/mm
dump=$1

workdir=$2
cd $workdir

sleep=60

for f in `curl "http://data.commoncrawl.org/crawl-data/CC-NEWS/$dump/warc.paths.gz" | gunzip --stdout`
do
    if [ ! -e $f ]
    then
        curl --create-dirs "http://data.commoncrawl.org/$f" -o $f.tmp
        if [ `stat --printf="%s" $f.tmp` -lt 100000 ]
        then
            echo "Throttled while downloading $f. Sleeping for $sleep seconds"
            rm $f.tmp
            sleep $sleep
            sleep=$((2*$sleep))
        else
            mv $f.tmp $f
            sleep=60
        fi
    fi
done

