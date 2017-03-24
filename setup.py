from setuptools import setup

setup(name='socialforcemodel',
      version='0.12',
      description='A python implementation of the Social Force Model',
      classifiers=[
          'Development Status :: 1 - Planning',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2.7'
      ],
      url='http://github.com/rexvalkering/socialforcemodel',
      author='Rex Valkering',
      author_email='rexvalkering@gmail.com',
      license='MIT',
      packages=['socialforcemodel'],
      install_requires=[
          'numpy',
          'matplotlib',
          'pyyaml'
      ],
      zip_safe=False)
