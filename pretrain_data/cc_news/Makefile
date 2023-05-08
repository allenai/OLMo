dump=2022/01
download-warc:
	download-warc.sh $(dump) /data/cc-news

maven:
	sudo apt-get install --yes maven

# See https://groups.google.com/g/common-crawl/c/hsb90GHq6to/m/SSVocyq8AAAJ
wet-processor: maven
	git clone https://github.com/commoncrawl/ia-web-commons
	cd ia-web-commons
	mvn install

	cd ..
	git clone https://github.com/commoncrawl/ia-hadoop-tools
	cd ia-hadoop-tools
	mvn package

wet: wet-processor
	mkdir -p /data/cc-news/$(dump)
	mkdir -p /data/cc-news/$(dump)/wat
	mkdir -p /data/cc-news/$(dump)/wet
	ln -s /data/cc-news/crawl-data/CC-NEWS/$(dump)  /data/cc-news/$(dump)/warc

	cd ia-hadoop-tools
	java -jar $PWD/target/ia-hadoop-tools-jar-with-dependencies.jar WEATGenerator -strictMode -skipExisting batch-id-xyz /data/cc-news/$(dump)/warc/*.warc.gz

upload-wet:
	aws s3 cp --recursive /data/cc-news/$(dump)/wet s3://ai2-llm/pretraining-data/sources/cc-news/raw/$(dump)
