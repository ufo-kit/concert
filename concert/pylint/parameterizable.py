import copy
from typing import TYPE_CHECKING

import astroid

import concert.base

if TYPE_CHECKING:
    from pylint.lint import PyLinter


def register(linter: "PyLinter") -> None:
    """This required method auto registers the checker during initialization.

    :param linter: The linter to register the checker to.
    """
    pass


def transform(cls):
    if cls.is_subtype_of("concert.base.Parameterizable"):
        cls_locals = copy.copy(cls.locals)
        for local in cls_locals:
            obj = cls.getattr(local)
            if isinstance(obj[0], astroid.AssignName):
                try:
                    cls['set_' + local] = [cls.instance_attr_ancestors('_set_' + local)]
                except:
                    pass

                try:
                    cls['set_' + local] = [cls.instance_attr_ancestors('_get_' + local)]
                except:
                    pass


astroid.MANAGER.register_transform(astroid.ClassDef, transform)
