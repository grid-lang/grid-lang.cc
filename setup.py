from setuptools import setup, find_packages

setup(
    name='gridlang',
    version='0.1',
    py_modules=['main', 'compiler', 'expression', 'array_handler', 'utils', 'scope',
                'control_flow', 'type_processor', 'parser', 'executor'],
    packages=find_packages(),
    install_requires=['pyarrow'],  # Add pyarrow dependency
    entry_points={
        'console_scripts': [
            'grid=main:main'  # Entry point to main function
        ]
    },
    author='GridLang Team',
    description='A Python-based tool for compiling and processing .grid files',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/grid-lang/grid-lang.cc',
    license='LGPLv3.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.7',
)
