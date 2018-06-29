#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='STG_Scoring',
    version='0.1',
    description='Package for automated scoring of reflections',
    url='http://github.com/JFLandrigan/STG_Scoring',
    author='Jon-Frederick Landrigan',
    author_email='jon.landrigan@gmail.com',
    packages = find_packages(),
    install_requires=['numpy', 'pandas', 'nltk', 'sklearn', 'psycopg2', 'string',
                    'enchant', 'gensim', 'collections', 'pickle'],
    zip_safe = False)