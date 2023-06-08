from setuptools import setup, find_packages

setup(
    name='au_dtw',
    version='0.2',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.22.1',
    ],
    description='Implementation of DTW, cDTW and FastDTW for use in a BSc thesis at the Department of Computer Science at Aarhus University, Denmark',
    url='',
    author='Anders Schroll Bjerregaard, Nicolai Landkildehus Lisle',
    author_email='201905192@post.au.dk, 202005828@post.au.dk',
    license='MIT',
)
