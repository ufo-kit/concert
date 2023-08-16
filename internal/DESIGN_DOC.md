# Concert Design Document

> Work In Progress

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
    <<base-class for objects featuring async properties>>
    +__new__()
}
note for AsyncObject "Realization of AsyncType. All concert objects
which need to be constructed asynchronously, need to extend this class."
AsyncObject ..|> AsyncType
```

## Parameters Object Structure

```mermaid
classDiagram

class AsyncObject {
    <<described in asynchronous base-types>>
}

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
    +Callable fget  # to read parameter
    +Callable fset  # to write parameter
    +Callable fget_target  # to get parameter target value
    # TODO - Understand more on the usage of parameters and associated attributes
    +setter_name() String
    +getter_name() String
    +getter_name_target() String
    +get_getter() Callable
    +get_target_getter() Callable
    +get_setter() Callable
}
note for Parameter "Properties associated with any concert object which might
need some. Parameters can check their states."

Parameterizable "1" --> "n" Parameter

class State {
    -_value(Device instance) String
}
note for State "Implements basic building blocks for state management, especially
applicable to devices, which may asynchronously transition from one state to another.
These transitions are relevant for us to acknowledge so that concert can decide on
the next step for the experiment."

State --|> Parameter

class Selection
note for Selection "Represents a parameter which can assume a value from pre-defined
list. Comparable to Enums."

Selection --|> Parameter

class Quantity {
    +UnitDefinition uint
    +
}
note for Quantity "Represents a parameter whose value denotes a measurable entity
having a unit."

Quantity --|> Parameter

class ParameterValue
ParameterValue --> Parameter

class StateValue
StateValue --|> ParameterValue

class QuantityValue
QuantityValue --|> ParameterValue

class SelectionValue
SelectionValue --|> ParameterValue
```

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

## Decentralized Implementation

The [PyZMQ](https://pyzmq.readthedocs.io/en/latest/#) makes the backbone of the decentralized networking stack for
concert. Concretely, we define a base class `ZmqBase` which makes the high-level API for a socket connection. On top
of that we define `ZmqSender` and `ZmqReceiver` as two abstract endpoints engaged in a peer-2-peer socket connection.
The `ZmqBase` class does not dictate the nature of the socket connection, rather delegates that implementation to
its derived classes. At the time of writing we have conceived two alternative connection paradigms, namely PUSH-PULL
oriented strong-coupling and PUB-SUB oriented loose coupling. Furthermore, we defined a `BroadcastReceiver` class
extending the `ZmqReceiver` as a listening endpoint from a source and propagate to all potentially interested parties
listed as the parameter _broadcast\_endpoints_ for the class. Following is the class diagram of the ZMQ implementation
stack.

```mermaid
classDiagram
class ZmqBase {
    <<base class for zmq image streams>>
    +String endpoint
    -Context context
    -Socket socket
    +connect(String endpoint)
    +close()
    -_setup_socket()
}
note for ZmqBase "base class for zmq-based image streams. Connects to a non-null endpoint if connection to that
endpoint is not established already, otherwise _setup_socket is triggered to establish a socket connection. The
implementation for establishing the socket connection must be provided by the derived class."

class ZmqSender {
    -Int _send_hwm
    -(override)_setup_socket()
    +send_image(Optional<NDArray> image)
}

note for ZmqSender "provides implementation for establishing a socket connection with high water mark. It means
specifying a hard limit on the max number of outstanding messages to be buffered in memory before sending to the
socket communication peer. Exposes the api for sending images, delegates to the zmq_send_image function."

ZmqSender --|> ZmqBase

class ZmqReceiver {
    -Int _rcv_hwm
    -Bool _reliable
    -String _topic
    -Poller _poller
    -Int _polling_timeout
    -Bool _request_stop
    -(override)_setup_socket()
    +stop()
    +close()
    +is_message_available(Optional~Int~ polling_timeout) Boolean
    +receive_message(Boolean return_metadata) RecvPayload_t
    +subscribe(Boolean return_metadata) AsyncIterable[Subscription_t]
}

note for ZmqReceiver "Creates a conditionally ZMQ_PULL | ZMQ_SUB type socket connection peer to receive images from
the ZmqSender endpoint. The socket connection type determines the degree of coupling with the sender endpoint."

ZmqReceiver --|> ZmqBase

class BroadcastServer {
    -Set~Socket~ _broadcast_sockets
    -Poller _poller_out
    -Event _finished
    -Boolean _request_stop_forwarding
    -_forward_image(Optional~NDArray~ image, Optional~Metadata~ metadata)
    +consume()
    +serve()
    +shutdown()
}

note for BroadcastServer "As the name suggests its main job is to listen for incoming data on its underlying receiver
endpoint and propagates same to all other interested parties who could use the data."

BroadcastServer --|> ZmqReceiver
```

### Orchestration of Tango devices

```mermaid
classDiagram
    class Device {
        <<Abstract Tango device server>>
    }
    note for Device "Each tango device is encapsulated by a so-called tango DeviceServer. In essence a
    DeviceServer represents a generic listener endpoint. We are responsible to provide implementation for various
    utilities that a concrete device server should serve."

    class TangoRemoteProcessing {
        <<Base device class that encapsulates a ZMQ receiver>>
        
    }
    note for TangoRemoteProcessing "Base implementation for a Tango device server. Encapsulates a ZmqReceiver instance,
    which is the actual consumer of the stream that is broadcast by the BroadcastReceiver."

    TangoRemoteProcessing ..|> Device
    
    class TangoBenchmarker {
        <<Abstract tango device for benchmarking>>
    }
    TangoBenchmarker ..|> TangoRemoteProcessing

    class TangoFileCamera {
        <<Abstract camera device to mock acquisitions reading from file system>>
    }
    TangoFileCamera ..|> TangoRemoteProcessing

    class TangoOnlineReconstruction {
        <<Abstract online reconstruction device to start reconstructing upon receiving each single acquisition>>
    }
    TangoOnlineReconstruction ..|> TangoRemoteProcessing

    class TangoWriter {
        <<Abstract storage handler device>>
    }
    TangoWriter ..|> TangoRemoteProcessing
```

```mermaid
sequenceDiagram
    TangoFileCamera (DeviceServer) ->> BroadcastReceiver: Acquisitions
    BroadcastReceiver ->> TangoOnlineReconstruction (DeviceServer): Acquisitions
    BroadcastReceiver ->> TangoWriter (DeviceServer): Acquisitions
    BroadcastReceiver ->> TangoBenchmarker (DeviceServer): Acquisitions
```
