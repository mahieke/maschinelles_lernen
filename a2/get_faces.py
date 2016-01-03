#! /usr/bin/env python
import urllib.request
import tarfile
import sys

def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))


def main():
    url = "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz"

    dest_file = "./lfw-funneled.tgz"

    # Download tar file
    print("Download tar file from {}".format(url))
    urllib.request.urlretrieve(url, dest_file, reporthook)

    print("Extract tar file ...")
    # extract tar file
    tar = tarfile.open(dest_file)
    tar.extractall('./')


if __name__ == '__main__':
    main()