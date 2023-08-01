# Concert Design Document

This document encapsulates the internal development structure for concert. It would depict the API and implementation
layer objects and their relationships using UML diagrams. In the process we'd skip the private and less-relevant
attributes e.g., logger from the structures. The `-` and `+` denotes private and public members of an object respectively.

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
    +exists(Set~String~ paths) bool
    +descend(String name) Walker
    +ascend() Walker
    +create_writer(AsyncGenerator producer, Optional~String~ name, Optional~String~ dsetname) Awaitable
}
note for DirectoryWalker "Moves through a file system and writes flat files using a specific filename template"
class DirectoryWalker {
    +(override) exists(Set~String~ paths) bool
    -_dset_exists(String dsetname) bool
}
Walker <|-- DirectoryWalker
```
