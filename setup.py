from setuptools import setup

setup(
    name='ultracov',
    version='0.1.0',
    description='tools for ultrasound video analysis',
    packages=['ultracov'],
    install_requires=[
        'PySimpleGUI==4.41.2',
        'opencv-python==4.5.2.52',
        'numpy==1.19.5',
        'numba==0.53.1',
        'scipy==1.6.3',
        'matplotlib==3.4.2',
        'Pillow==8.2.0',
        'tensorflow==2.5.0rc3',
        'scikit-image==0.18.1'
    ],
    include_package_data=True
)