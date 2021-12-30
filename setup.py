from setuptools import setup, find_packages
from pathlib import Path

__author__ = 'Haoyan Huo'
__maintainer__ = 'Haoyan Huo'
__email__ = 'hhaoyann@gmail.com'

here = Path(__file__).parent
long_description = (here / "README.md").read_text()

if __name__ == '__main__':
    setup(
        name='opt-einsum-torch',
        author='Haoyan Huo',
        author_email='hhaoyann@gmail.com',
        url='http://github.com/hhaoyan/opt-einsum-torch',
        description='Memory-efficient optimum einsum using opt_einsum planning '
                    'and PyTorch kernels.',
        classifiers=[
            'Environment :: GPU',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3 :: Only',
            'Development Status :: 1 - Planning',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Topic :: Software Development :: Libraries :: Python Modules'
        ],
        python_requires=">=3.6",
        license='MIT',
        include_package_data=True,
        version='0.1.0',
        packages=find_packages(),
        install_requires=open('requirements.txt').readlines(),
        long_description=long_description,
        long_description_content_type='text/markdown',
    )
