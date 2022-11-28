import setuptools

with open('README.md', 'r') as fh:
    README = fh.read()

VERSION = '0.1.3.1'

setuptools.setup(
    name='causal-learn',
    version=VERSION,
    author='',
    description='causal-learn Python Package',
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
        'pydot',
        'tqdm'
    ],
    url='https://github.com/cmu-phil/causal-learn',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License (MIT)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
