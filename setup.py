from setuptools import find_packages
from setuptools import setup

install_requires = ['numpy', 'matplotlib', 'cvxopt', 'scipy']
tests_require = ['pytest']
setup_requires = ["pytest-runner"]

setup(
    name='PythonLinearNonlinearControl',
    version='2.0',
    description='Implementing linear and nonlinear control method in python',
    author='Shunichi Sekiguchi',
    author_email='quick1st97of@gmail.com',
    install_requires=install_requires,
    url='https://github.com/Shunichi09/PythonLinearNonlinearControl',
    license='MIT License',
    packages=find_packages(exclude=('tests')),
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require
)