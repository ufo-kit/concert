# -*- coding: utf-8 -*-
import sys, os

sys.path.append(os.path.join(os.getcwd(), '..'))

extensions = ['sphinx.ext.autodoc']
# templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

project = u'bossa'
copyright = u'2013, Matthias Vogelgesang'

version = '0.0'
release = '0.0.1'

exclude_patterns = ['_build']
pygments_style = 'sphinx'

# html_static_path = ['_static']
htmlhelp_basename = 'bossadoc'


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
  ('index', 'bossa.tex', u'bossa Documentation',
   u'Matthias Vogelgesang', 'manual'),
]


# -- Options for manual page output --------------------------------------------
man_pages = [
    ('index', 'bossa', u'bossa Documentation',
     [u'Matthias Vogelgesang'], 1)
]
