# Data Structures Final Review

## Arrays and Linked Lists

Arrays store elements in contiguous memory, so random access by index is O(1).
Insertion in the middle is usually O(n) because later elements must move.
Linked lists store nodes separately and connect them with pointers.
They are good for frequent insertion or deletion when the target node is already known.

## Stacks and Queues

A stack follows Last In First Out. Common operations are push, pop, and peek.
Stacks are useful for function calls, expression parsing, undo history, and depth-first search.
A queue follows First In First Out. Common operations are enqueue and dequeue.
Queues are useful for task scheduling, breadth-first search, and buffering.

## Hash Tables

A hash table maps keys to array positions through a hash function.
Average lookup, insertion, and deletion are O(1), but collisions must be handled.
Common collision strategies include chaining and open addressing.
The load factor measures how full the table is and affects performance.

## Trees

A tree is a hierarchical structure made of nodes and edges.
Binary search trees keep smaller values on the left and larger values on the right.
Balanced trees reduce the height and keep operations close to O(log n).
Tree traversal orders include preorder, inorder, postorder, and level order.
