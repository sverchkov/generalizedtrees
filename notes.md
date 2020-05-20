# Development work notes

## Design of our tree data structure

### History

At v0.0.3 and prior, the underlying tree data structure was modeled as a collection of nodes.
Specifically, the following object types were used:

* `core.Node` with members
    * `parent` - another node or `None` if this is a root
    * `constraints` - constraints associated with the branch to the node
    * `model` - an object that implements a `predict` method.
        * The model is also used for linking to the children of a node using a specialized class,`core.ChildSelector`. A `ChildSelector`:
            * Holds an array of children
            * Implements the logic of checking each child's conditions when attempting to `predict`.

This design was elegant in its simplicity of definitions.
Particularly, defining prediction in terms or recursion simplifies implementation and makes it easy to implement model trees.

### Needs

In attempting to reimplement Trepan and other prior model explanation methods, it has become clear that there are certain needs not met by the v0.0.3 data strucutre.
These needs are:

* Ability to efficiently keep track of the number of nodes
* Querying depth
* Attaching auxillary information to nodes
    * In addition to the constraints on the path to the node
    * For Trepan:
        * Training set samples that reach the node
        * Pointer to sample generator
        * Fidelity and Reach estimates
* Ability to pretty-print the tree

We've considered the `anytree` package, the followind doesn't quite fit our use case:
* Querying depth/number of nodes is still a tree walk
* Nodes have names as text

### Tree data structure in v0.0.4

Going forward, we'll be using our own tree data structure, defined in the `tree` subpackage.