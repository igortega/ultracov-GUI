import ultracov
import os
from zipfile import ZipFile
from urllib.request import urlretrieve

here = os.path.split(ultracov.__file__)[0]

zip_path = os.path.join(here, 'required_files.zip')
if not os.path.exists(zip_path):
    # download required files
    url = r"http://x-cov.com/data/ULTRACOV_UCM_V0.zip"
    print('Downloading required files...')
    urlretrieve(url, zip_path)

    zfile = ZipFile(zip_path)
    # extract files to each directory
    directories = ['pleura', 'similarity']
    for d in directories:
        requirements_path = os.path.join(here, d, 'file_requirements.txt')
        with open(requirements_path) as f:
            files = f.readlines()
        for f in files:
            fname = f.split('\n')[0]
            zfile.extract(fname, path=os.path.join(here, d))

    zfile.close()
