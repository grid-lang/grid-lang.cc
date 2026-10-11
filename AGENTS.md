See index.md for project details. There is a update_index_lines.py to update line numbers, and the file should be updated after a commit.

The agent should only touch files inside the project folder. Create a local temp directory for temporary data, don't try to use /tmp.

The Grid language reads variable bindings in a declarative way, indifferent to order. A line with just "var = value" defines a binding. Effects run in order; Push is used to propagate a value in streaming fashion.
