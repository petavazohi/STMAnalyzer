from setuptools import setup, find_packages

setup(
    name="STMAnalyzer",  # Replace with your package name
    version="0.1.0",
    author="Pedram Tavadze",
    author_email="petavazohi@gmail.com",
    description="A brief description of your package",  # Update with a brief package description
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/petavazohi/STMAnalyzer",  # Replace with your project URL
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy > 1.20",
        "matplotlib > 3.7",
        "scikit-learn > 0.24",
        "napari > 0.4",
        "colorbar > 3.0",
        "matplotlib-scalebar > 0.7",
        "scikit-image > 0.18",
        'nanonispy',
    ],
)

# Instructions:
# 1. Replace "STMAnalyzer" with the name of your package.
# 2. Replace "Your Name" and "your.email@example.com" with your actual name and email address.
# 3. Provide a brief description of your package.
# 4. Make sure you have a valid "README.md" file in your directory.
# 5. Update the "url" field with the URL of your GitHub repository or other project URL.
# 6. List any package dependencies in the "install_requires" section.
# 7. Run `python setup.py sdist bdist_wheel` to build your package.
# 8. Use `twine upload dist/*` to upload your package to PyPI.
