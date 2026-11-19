# Configuration file for the Sphinx documentation builder.
#
# Full list of options: https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime

# -- Project information -----------------------------------------------------

from jstark import __version__

release = __version__
version = __version__

project = "jstark"
author = "Jayshil A. Patel"
copyright = f"{datetime.datetime.now().year}, {author}"  # noqa: A001

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "sphinx_copybutton",
    "nbsphinx",
]



# docs/MD holds internal development notes: not part of the website.
exclude_patterns = ["_build", "MD", "._*", "**/._*", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

source_suffix = {".rst": "restructuredtext"}

master_doc = "index"

# Treat everything in single ` as a Python reference.
default_role = "py:obj"

# -- autodoc / automodapi ----------------------------------------------------

# Append the __init__ docstring (constructor parameters) to the class docstring.
autoclass_content = "both"
autodoc_member_order = "bysource"
automodapi_toctreedirnm = "api"
numpydoc_show_class_members = False

# -- Notebooks (tutorials) ---------------------------------------------------

# The tutorials need tens of GB of JWST data and hours of computing: they are
# executed locally and stored with their outputs; the docs build never runs them.
nbsphinx_execute = "never"
nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) | string %}

.. note::

   This page was generated from a Jupyter notebook:
   :download:`{{ docname.split('/')[-1] }} </{{ docname }}>`. It was run
   locally on the full data sets; the outputs shown are from that run.
"""

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"jstark {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "github_url": "https://github.com/Jayshil/jstark",
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}
html_context = {"default_mode": "auto"}
html_show_sourcelink = False

# -- Other options -----------------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# Render "Attributes" docstring sections as fields, not as separate attribute
# entries (which automodapi documents again for dataclasses).
napoleon_use_ivar = True
