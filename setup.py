import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="keras-toolkit",
    version="0.1.0rc4",
    author="Xing Han Lu",
    author_email="github@xinghanlu.com",
    description="A collection of functions to help you easily train and run Tensorflow Keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xhlulu/keras-toolkit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=["tensorflow"],
    extras_require={"dev": ["pytest", "black", "jinja2"]},
)
