from setuptools import setup, find_packages

setup(
  name = 'tiger-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.1',
  license='MIT',
  description = 'Tiger Optimizer - Pytorch',
  author = 'MrSteyk',
  author_email = 'mrsteyk1@vk.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/mrsteyk/tiger-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'optimizers'
  ],
  install_requires=[
    'torch>=1.6'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.10',
  ],
)