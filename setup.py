from setuptools import setup, find_packages

setup(
    name='chatosaurus-embedder',
    version='0.1.0',
    description='Embedding generator for Chatosaurus documentation',
    author='The Dino Stack',
    license='MIT',
    packages=find_packages(include=['embed_providers', 'embed_providers.*', 'utils', 'utils.*']),
    install_requires=[],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'chatosaurus-embedder=main:main'
        ]
    },
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)