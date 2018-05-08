from setuptools import setup, find_packages

version = '0.1'

install_requires = [
    "numpy",
    "tensorflow",
    "keras",
    "pandas",
    "spacy",
    "tensorboard"
]

extras = {
    'dev': [
        "autopep8",
        "rope_py3k",
        "importmagic",
        "yapf"
    ]   
}

setup(
    name='tfidf_sts',
    version=version,
    description="Packages for tfidf analysis of STS.",
    long_description="",
    classifiers=[],  # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
    keywords='',
    author='',
    author_email='',
    url='',
    license='',
    packages=find_packages(exclude=['']),
    include_package_data=True,
    zip_safe=True,
    install_requires=install_requires,
    tests_require=tests_requires,
    extras_require=extras
)
