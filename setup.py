"""
RaceIQ Pro - Setup Configuration
Toyota GR Cup Hackathon Project

Package installation script for easy deployment and distribution.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f
                   if line.strip() and not line.startswith('#')]

# Development dependencies
dev_requirements = [
    'pytest>=7.4.3',
    'pytest-cov>=4.1.0',
    'black>=23.12.1',
    'flake8>=7.0.0',
    'mypy>=1.7.1',
    'isort>=5.13.2',
    'pre-commit>=3.6.0',
]

# Documentation dependencies
docs_requirements = [
    'sphinx>=7.2.6',
    'sphinx-rtd-theme>=2.0.0',
    'sphinx-autodoc-typehints>=1.25.2',
]

setup(
    # Package metadata
    name='raceiq-pro',
    version='1.0.0',
    description='Advanced Racing Intelligence Platform for Toyota GR Cup',
    long_description=long_description,
    long_description_content_type='text/markdown',

    # Author information
    author='Toyota GR Cup Hackathon Team',
    author_email='contact@raceiqpro.com',  # Update with actual email

    # URLs
    url='https://github.com/yourusername/ToyotaGR',  # Update with actual repo
    project_urls={
        'Bug Tracker': 'https://github.com/yourusername/ToyotaGR/issues',
        'Documentation': 'https://github.com/yourusername/ToyotaGR/blob/main/docs/IMPLEMENTATION.md',
        'Source Code': 'https://github.com/yourusername/ToyotaGR',
    },

    # Package discovery
    packages=find_packages(exclude=['tests', 'docs', 'notebooks', 'Research']),

    # Python version requirement
    python_requires='>=3.9',

    # Dependencies
    install_requires=requirements,

    # Optional dependencies
    extras_require={
        'dev': dev_requirements,
        'docs': docs_requirements,
        'all': dev_requirements + docs_requirements,
    },

    # Package data
    include_package_data=True,
    package_data={
        'raceiq': [
            'config/*.yaml',
            'config/*.json',
        ],
    },

    # Entry points for CLI commands
    entry_points={
        'console_scripts': [
            'raceiq=raceiq.cli:main',  # CLI interface (to be implemented)
            'raceiq-web=raceiq.app:main',  # Web interface launcher
        ],
    },

    # Classification
    classifiers=[
        # Development status
        'Development Status :: 4 - Beta',

        # Intended audience
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Other Audience',  # Racing teams

        # Topic
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',

        # License
        'License :: OSI Approved :: MIT License',

        # Python versions
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',

        # Operating systems
        'Operating System :: OS Independent',

        # Natural language
        'Natural Language :: English',
    ],

    # Keywords for PyPI search
    keywords=[
        'racing',
        'motorsports',
        'data-analysis',
        'machine-learning',
        'telemetry',
        'analytics',
        'toyota-gr-cup',
        'pit-strategy',
        'tire-degradation',
        'anomaly-detection',
        'streamlit',
    ],

    # License
    license='MIT',

    # Zip safe
    zip_safe=False,
)
