from setuptools import setup, find_packages

setup(
    name='myapp_rag_ft',
    version='0.1.0',
    package_dir={'': 'src'},  # Esto le indica a setuptools que busque los paquetes en 'src'
    packages=find_packages(where='src'),  # Encuentra todos los paquetes bajo 'src'
    install_requires=[],
)
