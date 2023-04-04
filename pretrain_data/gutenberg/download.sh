#!/bin/bash
#
# Download the complete archive of text format files from Project Gutenberg.
#
# Estimated size in Q2 2014: 7G in zipfiles which unzip to about 21G in text
# files. So have 30G spare if you run this.
#
# Note that as written here this is a 36 to 48 hour process on a fast
# connection, with pauses between downloads. This minimizes impact on the
# Project Gutenberg servers.
#
# You'll only have to do this once, however, and this script will pick up from
# where it left off if it fails or is stopped.
#

# ------------------------------------------------------------------------
# Preliminaries
# ------------------------------------------------------------------------

set -o nounset
set -o errexit

# Restrict downloads to this file format.
FORMAT=txt

# The directory in which this file exists or the one provided as the first
# argument.
if [ $# -eq 0 ] ; then
  DIR="$( cd "$( dirname "$0" )" && pwd)"
else
  DIR="$1"
fi

# File containing the list of zipfile URLs.
ZIP_LIST="${DIR}/zipfileLinks.txt"
# A subdirectory in which to store the zipfiles.
ZIP_DIR="${DIR}/zipfiles"
# A directory in which to store the unzipped files.
UNZIP_DIR="${DIR}/files"

mkdir -p "${ZIP_DIR}"
mkdir -p "${UNZIP_DIR}"

# ------------------------------------------------------------------------
# Obtain URLs to download.
# ------------------------------------------------------------------------

# This step downloads ~700 html files containing ~38,000 zip file links. This
# will take about 30 minutes.

echo "-------------------------------------------------------------------------"
echo "Harvesting zipfile URLs for format [${FORMAT}]"
echo "-------------------------------------------------------------------------"

# Only do this if it hasn't been done already.
if [ ! -f "${ZIP_LIST}" ] ; then
  # The --mirror mode of wget spiders through files listing links.
  # The two second delay is to play nice and not get banned.
  wget \
    --wait=2 \
    --mirror \
    "http://www.gutenberg.org/robot/harvest?filetypes[]=${FORMAT}"

  # Process the downloaded HTML link lists into a single sorted file of zipfile
  # URLs, one per line.
  grep -oh 'http://[a-zA-Z0-9./]*.zip' "${DIR}/www.gutenberg.org/robot/harvest"* | \
    sort | \
    uniq > "${ZIP_LIST}"

  # Get rid of the downloaded harvest files now that we have what we want.
  rm -Rf "${DIR}/www.gutenberg.org"
else
  echo "${ZIP_LIST} already exists. Skipping harvest."
fi

# ------------------------------------------------------------------------
# Download the zipfiles.
# ------------------------------------------------------------------------

# This will take a while: 36 to 48 hours. Just let it run. Project Gutenberg is
# a non-profit with a noble goal, so don't crush their servers, and it isn't as
# though you'll need to do this more than once.

echo "-------------------------------------------------------------------------"
echo "Downloading zipfiles."
echo "This will take 36-48 hours if starting from scratch."
echo "-------------------------------------------------------------------------"

for URL in $(cat "${ZIP_LIST}")
do
  ZIP_FILE="${ZIP_DIR}/${URL##*/}"
  # Only download it if it hasn't already been downloaded in a past run.
  if [ ! -f "${ZIP_FILE}" ] ; then
    wget --directory-prefix="${ZIP_DIR}" "${URL}"
    # Play nice with a delay.
    sleep 2
  else
    echo "${ZIP_FILE##*/} already exists. Skipping download."
  fi
done

# ------------------------------------------------------------------------
# Unzip the zipfiles.
# ------------------------------------------------------------------------

echo "-------------------------------------------------------------------------"
echo "Unzipping files."
echo "-------------------------------------------------------------------------"

for ZIP_FILE in $(find ${ZIP_DIR} -name '*.zip')
do
  UNZIP_FILE=$(basename ${ZIP_FILE} .zip)
  UNZIP_FILE="${UNZIP_DIR}/${UNZIP_FILE}.txt"
  # Only unzip if not already unzipped. This check assumes that x.zip unzips to
  # x.txt, which so far seems to be the case.
  if [ ! -f "${UNZIP_FILE}" ] ; then
    unzip -o "${ZIP_FILE}" -d "${UNZIP_DIR}"
  else
    echo "${ZIP_FILE##*/} already unzipped. Skipping."
  fi
done
