# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from concert import __version__

_authors = [u'Matthias Vogelgesang',
            u'Tomas Farago',
            u'Marcus Zuber']

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.intersphinx',
              'sphinx.ext.mathjax',
              'sphinxcontrib_trio']

source_suffix = '.rst'
master_doc = 'index'

project = u'Concert'
copyright = u'2013, ' + u', '.join(_authors)

version = __version__[:__version__.rfind('.')]
release = __version__

exclude_patterns = ['_build']
pygments_style = 'sphinx'

autoclass_content = 'both'

autodoc_default_options = {
    'members': True,
    'special-members': '__ainit__'
}

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_style = 'css/custom.css'
htmlhelp_basename = 'concertdoc'

intersphinx_mapping = {
    'python': ('http://python.readthedocs.org/en/latest/', None),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', None)
}

# -- Options for LaTeX output --------------------------------------------------
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',
}

latex_documents = [
    ('index', 'concert.tex', u'Concert Documentation',
     u', '.join(author.replace(' ', '~') for author in _authors),
     'manual'),
]

man_pages = [
    ('index', 'concert', u'Concert Documentation',
     _authors, 1)
]
