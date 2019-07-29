import tarfile


def extract(filename):
    print('Extracting {}...'.format(filename))
    with tarfile.open(filename) as tar:
        tar.extractall()


if __name__ == "__main__":
    extract('data/mjsynth.tar.gz')
