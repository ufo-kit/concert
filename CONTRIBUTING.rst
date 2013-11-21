Reporting bugs
--------------

Any bugs concerning the Concert core library and script should be reported as an
issue on the GitHub `issue tracker`_.

.. _issue tracker: https://github.com/ufo-kit/concert/issues


Fixing bugs or adding features
------------------------------

Bug fixes and new features **must** be in `pull request`_ form. Pull request
commits should consist of single logical changes and bear a clear message
respecting common commit message `conventions`_. Before the change is merged
eventually it must be rebased against master.

Bug fixes must come with a unit test that will fail on the bug and pass with the
fix. If an issue exists reference it in the branch name and commit message, e.g.
``fix-92-remove-foo`` and "Fix #92: Remove foo".

New features **must** follow `PEP 8`_ and must be documented thoroughly.

.. _pull request: https://github.com/ufo-kit/concert/pulls
.. _conventions: http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html
.. _PEP 8: http://www.python.org/dev/peps/pep-0008/
