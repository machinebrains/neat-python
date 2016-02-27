from setuptools import setup

setup(
    name='neatsociety',
    version='0.1',
    author='cesar.gomes, mirrorballu2,amro-pydev',
    author_email='amrsfmt@yahoo.com',
    maintainer='Amro Tork',
    maintainer_email='amrsfmt@yahoo.com',
    url='https://github.com/machinebrains/neat-society.git',
    license="BSD",
    description='NEAT (NeuroEvolution of Augmenting Topologies) Society Algoritm implementation',
    long_description='Python implementation of NEAT (NeuroEvolution of Augmenting Topologies) society algorithm.',
    packages=['neatsociety', 'neatsociety/iznn', 'neatsociety/nn', 'neatsociety/ctrnn', 'neatsociety/ifnn'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Topic :: Scientific/Engineering'
    ]
)
