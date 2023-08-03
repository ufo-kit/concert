# Concert Design Document

This document encapsulates the internal development structure for concert. It would depict the API and implementation
layer objects and their relationships using UML diagrams. In the process we'd skip the private and less-relevant
attributes e.g., logger from the structures. The `-` and `+` denotes private and public members of an object respectively.

## Asynchronous Base Types

```mermaid
classDiagram
class AsyncType {
    <<meta-class for asynchronous construction>>
    +__new__()
    +__call__()
}
note for AsyncType "Meta-class which allows an awaitable constructor.
This class provides implementation of the __call__ and __new__ method
to dictate that any class which use AsyncType as the meta-type has to
provide an awaitable constructor instead of the normal one. This class
is not meant to be subclassed directly."

class AsyncObject {
    +__new__()
}
note for AsyncObject "Realization of AsyncType. All concert objects
which need to be constructed asynchronously, need to extend this class."
AsyncObject ..|> AsyncType

class Parameterizable {
    +__iter__() Parameter
    +__getitem__(String param) Parameter
    +__contains__(String key) Boolean
    +info_table() TableType
    +install_parameter(Dict~String->Parameter~)
    +stash()
    +restore()
    +lock(Boolean permanent)
    +unlock()
}
note for Parameterizable "Collection of parameters. Any concert object
which might depend on a set of parameters to function in their intended
way such as a device (e.g., camera), a utility (e.g., GeneralBackProjectManager)
or even concert experiment entity (e.g., Acquisition) could extend Parameterizable
to represent the idea of having a set of parameters. It is a generator object of
the parameters which can be used as an iterable."
Parameterizable --|> AsyncObject

class Parameter {
    +setter_name() String
    +getter_name() String
    +getter_name_target() String
    +get_getter() Callable
    +get_target_getter() Callable
    +get_setter() Callable
}
note for Parameter "Properties associated with any concert object which might
need some. Parameters can check their states"

Parameterizable "1" --> "n" Parameter
```

## Device Object Structures

> TODO

## File System Traversal

```mermaid
classDiagram
class Walker {
    <<base-class for directory traversal>>
    -String _root
    -String _current
    +String dsetname
    +home()
    +current() String
    +exists(Set~String~ paths) Boolean
    +descend(String name) Walker
    +ascend() Walker
    +create_writer(AsyncIterable~ArrayLike~ producer, Optional~String~ name, Optional~String~ dsetname) Awaitable
}

class DirectoryWalker {
    +(override) exists(Set~String~ paths) Boolean
    -_dset_exists(String dsetname) Boolean
}
note for DirectoryWalker "Moves through a file system and writes flat files using a specific filename template"
Walker <|-- DirectoryWalker
```