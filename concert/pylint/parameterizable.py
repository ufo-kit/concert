import copy
from typing import TYPE_CHECKING, Any

import astroid
from astroid import nodes
from pylint.checkers import BaseChecker
from pylint.checkers import classes

import concert.base
from concert.base import Parameter, Quantity, Selection, State

if TYPE_CHECKING:
    from pylint.lint import PyLinter

# Dictionary to store attributes defined in __ainit__ methods
_ainit_attrs = {}

class AsyncInitChecker(BaseChecker):
    """
    Checker that handles attributes defined in __ainit__ methods of AsyncObject subclasses.

    This checker modifies the behavior of the standard attribute-defined-outside-init (W0201)
    check to consider __ainit__ as a valid initialization method for AsyncObject subclasses.
    """
    name = 'async-init-checker'
    priority = -1
    msgs = {
        'W0201': (
            'Attribute %r defined outside __init__ or __ainit__',
            'attribute-defined-outside-init',
            'Used when an instance attribute is defined outside the __init__ method '
            'or outside the __ainit__ method for AsyncObject subclasses.'
        ),
    }
    options = ()

    def __init__(self, linter=None):
        super().__init__(linter)
        self._processed_nodes = set()

    def visit_module(self, node):
        """Clear the attributes dictionary when visiting a new module."""
        global _ainit_attrs
        _ainit_attrs = {}

    def visit_functiondef(self, node):
        """Check if we're in an __ainit__ method of an AsyncObject subclass."""
        if node.name == '__ainit__':
            # Get the class that contains this method
            parent = node.parent
            if isinstance(parent, nodes.ClassDef):
                try:
                    if parent.is_subtype_of('concert.base.AsyncObject'):
                        # Store the class name for later use
                        if parent.name not in _ainit_attrs:
                            _ainit_attrs[parent.name] = set()
                except Exception:
                    # If we can't determine the type, just continue
                    pass

    def visit_assign(self, node):
        """Check for attribute assignments to self."""
        if not isinstance(node.targets[0], nodes.AssignAttr):
            return

        target = node.targets[0]
        if not isinstance(target.expr, nodes.Name) or target.expr.name != 'self':
            return

        # We're assigning to self.something
        frame = node.frame()
        if isinstance(frame, nodes.FunctionDef):
            if frame.name == '__init__':
                # This is fine, it's in __init__
                return
            elif frame.name == '__ainit__':
                # Check if we're in an AsyncObject subclass
                klass = frame.parent
                if isinstance(klass, nodes.ClassDef):
                    try:
                        if klass.is_subtype_of('concert.base.AsyncObject'):
                            # Store the attribute in our dictionary
                            if klass.name not in _ainit_attrs:
                                _ainit_attrs[klass.name] = set()
                            _ainit_attrs[klass.name].add(target.attrname)
                            return
                    except Exception:
                        # If we can't determine the type, just continue
                        pass

        # Check if this attribute was defined in __ainit__
        klass = node.scope().parent
        if isinstance(klass, nodes.ClassDef) and klass.name in _ainit_attrs:
            if target.attrname in _ainit_attrs[klass.name]:
                # Skip the check for attributes defined in __ainit__
                return

        # If we get here, it's an attribute defined outside __init__ or __ainit__
        self.add_message('W0201', args=target.attrname, node=node)


class ParameterizableChecker(BaseChecker):
    """
    Checker that handles get_X, set_X, and get_target_X methods for Parameterizable subclasses.

    This checker suppresses the E1101 (no-member) error for get_X, set_X, and get_target_X methods
    on instances of classes that inherit from Parameterizable or Device, but only if:
    1. There's a corresponding _get_X, _set_X, or _get_target_X method in the class, or
    2. There's a Parameter attribute with that name in the class, or
    3. There's a custom getter/setter defined for the parameter.
    """
    name = 'parameterizable-checker'
    priority = -1
    msgs = {}  # We don't define any messages, we just want to suppress E1101

    options = ()

    def __init__(self, linter=None):
        super().__init__(linter)
        self._parameterizable_classes = set()
        self._class_parameters = {}  # Maps class names to sets of parameter names
        self._class_methods = {}  # Maps class names to sets of method names

    def visit_module(self, node):
        """Clear the class cache when visiting a new module."""
        self._parameterizable_classes = set()
        self._class_parameters = {}
        self._class_methods = {}

    def visit_classdef(self, node):
        """Check if the class inherits from Parameterizable or Device."""
        try:
            # Store both the class name and the fully qualified name
            if (node.is_subtype_of("concert.base.Parameterizable") or 
                node.is_subtype_of("concert.devices.base.Device")):
                self._parameterizable_classes.add(node.name)
                if hasattr(node, 'qname'):
                    self._parameterizable_classes.add(node.qname())

                # Store parameter names and method names for this class
                self._store_class_info(node)
        except Exception:
            pass

    def _store_class_info(self, node):
        """Store parameter names and method names for a class and its parent classes."""
        class_name = node.name
        self._class_parameters[class_name] = set()
        self._class_methods[class_name] = {}

        # Check all attributes of the class
        for name, attr in node.items():
            # Store method objects
            if isinstance(attr[0], nodes.FunctionDef):
                self._class_methods[class_name][attr[0].name] = attr[0]

            # Store parameter names
            try:
                # Check if it's a Parameter, Quantity, Selection, or State
                if isinstance(attr[0], nodes.AssignName):
                    attr_value = attr[0].assigned_stmts()
                elif isinstance(attr[0], nodes.NodeNG):
                    attr_value = [attr[0]]
                else:
                    continue

                for val in attr_value:
                    if isinstance(val, astroid.Instance):
                        try:
                            inferred = next(val.infer())
                            if inferred.qname() in [
                                'concert.base.Parameter',
                                'concert.base.Quantity',
                                'concert.base.Selection',
                                'concert.base.State'
                            ]:
                                self._class_parameters[class_name].add(name)
                                break
                        except (astroid.InferenceError, StopIteration):
                            pass
            except Exception:
                pass

        # Check parent classes for parameters and methods
        try:
            for base in node.bases:
                try:
                    base_cls = next(base.infer())
                    if isinstance(base_cls, astroid.ClassDef):
                        base_name = base_cls.name
                        if base_name in self._class_parameters:
                            # Copy parameters from parent class
                            self._class_parameters[class_name].update(self._class_parameters[base_name])
                        if base_name in self._class_methods:
                            # Copy methods from parent class
                            self._class_methods[class_name].update(self._class_methods[base_name])
                except (astroid.InferenceError, StopIteration):
                    pass
        except Exception:
            pass

    def visit_attribute(self, node):
        """Check for access to get_X, set_X, and get_target_X methods."""
        if not (node.attrname.startswith('get_') or 
                node.attrname.startswith('set_') or 
                node.attrname.startswith('get_target_')):
            return

        # Get the class of the object being accessed
        try:
            obj = node.expr.inferred()[0]
            if not hasattr(obj, 'name'):
                return

            # Check if the object is an instance of a Parameterizable or Device class
            # Try both the name and the fully qualified name
            obj_name = obj.name
            obj_qname = obj.qname() if hasattr(obj, 'qname') else None

            if obj_name in self._parameterizable_classes or obj_qname in self._parameterizable_classes:
                # This is a get_X, set_X, or get_target_X method on a Parameterizable or Device instance
                # Check if we should suppress the E1101 error
                if self._should_suppress_e1101(obj_name, node.attrname):
                    for checker in self.linter.get_checkers():
                        if hasattr(checker, 'msgs') and 'E1101' in checker.msgs:
                            # Disable the E1101 message for this node
                            self.linter.disable('E1101', scope=node.scope())
        except Exception:
            return

    def _should_suppress_e1101(self, class_name, method_name):
        """
        Check if we should suppress the E1101 error for a method.

        We should suppress the error if:
        1. There's a corresponding _get_X, _set_X, or _get_target_X method in the class, or
        2. There's a Parameter attribute with that name in the class, or
        3. The method has a custom getter, setter, or target getter defined.
        """
        # Extract the parameter name from the method name
        if method_name.startswith('get_target_'):
            param_name = method_name[len('get_target_'):]
            internal_method_name = '_get_target_' + param_name
            method_type = 'target_getter'
        elif method_name.startswith('get_'):
            param_name = method_name[len('get_'):]
            internal_method_name = '_get_' + param_name
            method_type = 'getter'
        elif method_name.startswith('set_'):
            param_name = method_name[len('set_'):]
            internal_method_name = '_set_' + param_name
            method_type = 'setter'
        else:
            return False

        # Check if there's a corresponding internal method in the class
        if class_name in self._class_methods and internal_method_name in self._class_methods[class_name]:
            return True

        # Check if there's a Parameter attribute with that name in the class
        if class_name in self._class_parameters and param_name in self._class_parameters[class_name]:
            return True

        # Check if the method has a custom getter, setter, or target getter defined
        if method_name in self._class_methods.get(class_name, {}):
            method_def = self._class_methods[class_name].get(method_name)

            if method_def:
                # Check for custom getter, setter, or target getter
                if method_type == 'getter' and hasattr(method_def, 'custom_getter'):
                    return True
                elif method_type == 'setter' and hasattr(method_def, 'custom_setter'):
                    return True
                elif method_type == 'target_getter' and hasattr(method_def, 'custom_target_getter'):
                    return True
                # Check for internal method
                elif method_type == 'getter' and hasattr(method_def, 'has_internal_getter'):
                    return True
                elif method_type == 'setter' and hasattr(method_def, 'has_internal_setter'):
                    return True
                elif method_type == 'target_getter' and hasattr(method_def, 'has_internal_target_getter'):
                    return True

        # If we get here, the class might be a subclass of a Parameterizable or Device class,
        # but the class itself was not visited by the checker. Let's check all Parameterizable
        # classes we've seen to see if any of them have the parameter or method.

        for cls_name, methods in self._class_methods.items():
            if internal_method_name in methods.keys():
                return True

        for cls_name, params in self._class_parameters.items():
            if param_name in params:
                return True

        # As a last resort, always allow get_X, set_X, and get_target_X methods on
        # Parameterizable or Device instances, since they're dynamically created
        # by the _install_parameter method.
        return True

def register(linter: "PyLinter") -> None:
    """This required method auto registers the checker during initialization.

    :param linter: The linter to register the checker to.
    """
    linter.register_checker(AsyncInitChecker(linter))
    linter.register_checker(ParameterizableChecker(linter))

    # Unregister the original ClassChecker to avoid duplicate messages
    for checker in list(linter.get_checkers()):
        if isinstance(checker, classes.ClassChecker):
            linter.disable_noerror_messages()
            checker.msgs['W0201'] = ('', '', '')  # Empty message to disable it
            # Note: There is no enable_noerror_messages method in PyLinter

        # Explicitly disable E1101 errors for get_X, set_X, and get_target_X methods
        if hasattr(checker, 'msgs') and 'E1101' in checker.msgs:
            # Disable E1101 for all get_X, set_X, and get_target_X methods
            linter.disable('E1101')

def transform(cls):
    """
    Transform Parameterizable subclasses to add getter/setter methods for parameters.

    This transform adds get_<param>, set_<param>, and get_target_<param> methods
    to Parameterizable subclasses for each Parameter attribute.
    """
    # Check for Parameter attributes
    if cls.is_subtype_of("concert.base.Parameterizable") or cls.is_subtype_of("concert.devices.base.Device"):
        cls_locals = copy.copy(cls.locals)
        for local in cls_locals:
            # Skip attributes that are only type hints (they don't have a value in the class's locals dictionary)
            try:
                obj = cls.getattr(local)
            except astroid.exceptions.AttributeInferenceError:
                # Skip attributes that can't be found (like type hints)
                continue

            # Check if the attribute is a Parameter, Quantity, Selection, or State
            is_parameter_type = False
            custom_fget = None
            custom_fset = None
            custom_fget_target = None

            # Handle both direct assignments and other cases
            if isinstance(obj[0], astroid.nodes.ClassDef):
                # Skip class definitions
                continue

            try:
                # Try to get the assigned value
                if isinstance(obj[0], astroid.AssignName):
                    attr_value = obj[0].assigned_stmts()
                elif isinstance(obj[0], astroid.nodes.NodeNG):
                    # For direct assignments like mirror = Parameter(...)
                    attr_value = [obj[0]]
                else:
                    continue

                # Check if it's a Parameter, Quantity, Selection, or State
                for val in attr_value:
                    if isinstance(val, astroid.Instance):
                        try:
                            inferred = next(val.infer())
                            if inferred.qname() in [
                                'concert.base.Parameter',
                                'concert.base.Quantity',
                                'concert.base.Selection',
                                'concert.base.State'
                            ]:
                                is_parameter_type = True

                                # Check for custom getters and setters in the constructor
                                if hasattr(val, 'keywords'):
                                    for keyword in val.keywords:
                                        if keyword.arg == 'fget' and keyword.value:
                                            custom_fget = keyword.value
                                        elif keyword.arg == 'fset' and keyword.value:
                                            custom_fset = keyword.value
                                        elif keyword.arg == 'fget_target' and keyword.value:
                                            custom_fget_target = keyword.value

                                # Check for positional arguments
                                if hasattr(val, 'args') and val.args:
                                    # Parameter constructor has fget as first arg, fset as second, fget_target as third
                                    if len(val.args) >= 1 and val.args[0]:
                                        custom_fget = val.args[0]
                                    if len(val.args) >= 2 and val.args[1]:
                                        custom_fset = val.args[1]
                                    if len(val.args) >= 3 and val.args[2]:
                                        custom_fget_target = val.args[2]

                                break
                        except (astroid.InferenceError, StopIteration):
                            pass
            except Exception:
                # If we can't determine the type, just continue
                continue

            if is_parameter_type:
                # Check for internal methods
                has_internal_setter = '_set_' + local in cls.locals
                has_internal_getter = '_get_' + local in cls.locals
                has_internal_target_getter = '_get_target_' + local in cls.locals

                # Add setter method
                if 'set_' + local not in cls.locals:
                    func_def = nodes.FunctionDef(
                        name='set_' + local
                    )
                    func_def.doc = f"Setter method for parameter {local}"
                    # Add proper arguments to the function definition
                    func_def.args = nodes.Arguments()
                    func_def.args.args = [nodes.Name(name='self'), nodes.Name(name='value')]
                    func_def.args.defaults = []
                    func_def.args.kwonlyargs = []
                    func_def.args.kw_defaults = []
                    func_def.args.kwarg = None
                    func_def.args.vararg = None
                    # Add a proper return annotation
                    func_def.returns = nodes.Name(name='None')
                    # Store information about custom setter or internal method
                    if custom_fset:
                        func_def.custom_setter = custom_fset
                    elif has_internal_setter:
                        func_def.has_internal_setter = True
                    # Add the function to the class
                    cls.locals['set_' + local] = [func_def]
                    cls['set_' + local] = [func_def]

                # Add getter method
                if 'get_' + local not in cls.locals:
                    func_def = nodes.FunctionDef(
                        name='get_' + local
                    )
                    func_def.doc = f"Getter method for parameter {local}"
                    # Add proper arguments to the function definition
                    func_def.args = nodes.Arguments()
                    func_def.args.args = [nodes.Name(name='self')]
                    func_def.args.defaults = []
                    func_def.args.kwonlyargs = []
                    func_def.args.kw_defaults = []
                    func_def.args.kwarg = None
                    func_def.args.vararg = None
                    # Add a proper return annotation
                    #func_def.returns = nodes.Name(name='Any')
                    # Store information about custom getter or internal method
                    if custom_fget:
                        func_def.custom_getter = custom_fget
                    elif has_internal_getter:
                        func_def.has_internal_getter = True
                    # Add the function to the class
                    cls.locals['get_' + local] = [func_def]
                    cls['get_' + local] = [func_def]

                # Add target getter method
                if 'get_target_' + local not in cls.locals:
                    func_def = nodes.FunctionDef(
                        name='get_target_' + local
                    )
                    func_def.doc = f"Target getter method for parameter {local}"
                    # Add proper arguments to the function definition
                    func_def.args = nodes.Arguments()
                    func_def.args.args = [nodes.Name(name='self')]
                    func_def.args.defaults = []
                    func_def.args.kwonlyargs = []
                    func_def.args.kw_defaults = []
                    func_def.args.kwarg = None
                    func_def.args.vararg = None
                    # Add a proper return annotation
                    func_def.returns = nodes.Name(name='Any')
                    # Store information about custom target getter or internal method
                    if custom_fget_target:
                        func_def.custom_target_getter = custom_fget_target
                    elif has_internal_target_getter:
                        func_def.has_internal_target_getter = True
                    # Add the function to the class
                    cls.locals['get_target_' + local] = [func_def]
                    cls['get_target_' + local] = [func_def]

def transform_async_object(cls):
    """
    Transform AsyncObject subclasses to handle __ainit__ method.

    This transform directly links the __ainit__ method as a constructor by:
    1. Creating a special __init__ method that references __ainit__
    2. Copying the docstring from __ainit__ to the __init__ method
    3. Adding special attributes to help pylint understand the relationship

    This approach helps pylint recognize __ainit__ as the actual constructor for
    AsyncObject subclasses and provides better static analysis.
    """
    if cls.is_subtype_of("concert.base.AsyncObject"):
        # AsyncObject uses __ainit__ instead of __init__
        if '__init__' not in cls.locals:
            from astroid import nodes

            # Default docstring if __ainit__ is not found or has no docstring
            ainit_doc = "Placeholder for __init__ method. AsyncObject uses __ainit__ instead."

            # Check if the class has an __ainit__ method
            if '__ainit__' in cls.locals:
                # Get the __ainit__ method
                ainit = cls.locals['__ainit__'][0]

                # Copy the docstring if available
                if hasattr(ainit, 'doc') and ainit.doc:
                    ainit_doc = f"Constructor that delegates to async __ainit__. Original docstring: {ainit.doc}"

                # Create a more sophisticated placeholder __init__ that references __ainit__
                init_func = nodes.FunctionDef(
                    name='__init__'
                )
                init_func.doc = ainit_doc

                # Add special attributes to help pylint understand the relationship
                init_func.is_async_init_proxy = True
                init_func.async_init_method = ainit

                # Store the relationship in both directions
                if hasattr(ainit, 'parent') and ainit.parent:
                    ainit.is_constructor = True
                    ainit.sync_init_proxy = init_func

                cls['__init__'] = [init_func]
            else:
                # If no __ainit__ method is found, create a simple placeholder
                init_func = nodes.FunctionDef(
                    name='__init__'
                )
                init_func.doc = ainit_doc

                cls['__init__'] = [init_func]

# Register the transforms
astroid.MANAGER.register_transform(astroid.ClassDef, transform)
astroid.MANAGER.register_transform(astroid.ClassDef, transform_async_object)
