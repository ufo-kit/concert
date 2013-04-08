# -*- coding: utf-8 -*-
import sys, os

_authors = [u'Matthias Vogelgesang',
            u'Tomas Farago']

sys.path.insert(0, os.path.abspath('..'))

extensions = ['sphinx.ext.autodoc']
source_suffix = '.rst'
master_doc = 'index'

project = u'Concert'
copyright = u'2013, ' + u', '.join(_authors)

version = '0.0'
release = '0.0.1'

exclude_patterns = ['_build']
pygments_style = 'sphinx'

# html_static_path = ['_static']
htmlhelp_basename = 'concertdoc'


# -- Options for LaTeX output --------------------------------------------------
latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
#'preamble': '',
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
