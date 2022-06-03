import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(name='vidio',
                 version='0.0.4',
                 author='Jim Bohnslav',
                 author_email='jbohnslav@gmail.com',
                 description='Video file IO',
                 long_description=long_description,
                 long_description_content_type='text/markdown',
                 include_package_data=True,
                 packages=setuptools.find_packages(),
                 classifiers=[
                     'Programming Language :: Python :: 3',
                     'Operating System :: OS Independent'
                 ],
                 python_requires='>=3.6',
                 install_requires=['h5py', 'numpy', 'opencv-python-headless'])
