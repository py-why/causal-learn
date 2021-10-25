import setuptools

with open('README.md', 'r') as fh:
    README = fh.read()

VERSION = '0.1.1.5'

setuptools.setup(
    name='Pytrad',
    version=VERSION,
    author='',
    description='Pytrad Python Package',
    long_description=README,
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'graphviz',
        'statsmodels',
        'pandas',
        'matplotlib',
        'networkx',
        'pydot'
    ],
    url='https://github.com/cmu-phil/pytrad',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)