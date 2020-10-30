import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="expectation_reflection", # Replace with your own username
    version="0.0.9",
    author="Danh-Tai HOANG",
    author_email="hoangdanhtai@gmail.com",
    description="Expectation Reflection for classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danhtaihoang/expectation_reflection",
    #packages=setuptools.find_packages(),
    packages=['expectation_reflection'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license='MIT',
    keywords=['machine learning','classification','network inference','statistics'],
    python_requires='>=3.6',
)


