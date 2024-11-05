from pathlib import Path
import sys

# Use pathlib to get the absolute path
sys.path.insert(0, str(Path('..').resolve()))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'STMAnalyzer'
copyright = '2024, Pedram Tavadze'
author = 'Pedram Tavadze'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'autoapi.extension',
]
autoapi_dirs = [
    Path(__file__).resolve().parent.parent.parent / 'STMAnalyzer',
    Path(__file__).resolve().parent.parent.parent / 'scripts'
]
print(autoapi_dirs)
templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_theme_options = {
    "github_url": "https://github.com/petavazohi/STMAnalyzer",
    "logo": {
        "text": "STMAnalyzer",
    },
    "navbar_end": ["search-field.html", "navbar-icon-links"],
}