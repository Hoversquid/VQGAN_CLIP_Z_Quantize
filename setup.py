import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="VQGAN_CLIP_Z_Quantize",
    version="0.0.1",
    author="Hoversquid",
    author_email="contactprojectworldmap@gmail.com",
    description="Creates images from phrases and image input with VQGAN and CLIP (Z Quantize Method)",
    long_description=long_description,
    url="https://github.com/Hoversquid/VQGAN_CLIP_Z_Quantize/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
