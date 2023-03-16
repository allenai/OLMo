# todo: parallel needs to be cited.
echo $(wc -l preparation/stack/small_urls.txt)
cat preparation/stack/small_urls.txt| parallel -j+0 --bar --max-args=1 python preparation/stack/download_url.py {1} stack