# todo: parallel needs to be cited.
echo $(wc -l urls.txt)
cat urls.txt| parallel -j 10 --bar --max-args=1 ./download_url.sh {1} stack
