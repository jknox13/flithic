# Authors: Joseph Knox josephk@allenistitute.org
# License:

from collections import deque

class Node(object):
    """..."""
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

def _in_order(root, result=[]):
    """..."""
    if root:
        if root.left:
            for node in _in_order(root.left):
                yield node

        yield root.data

        if root.right:
            for node in _in_order(root.right):
                yield node

def _pre_order(root):
    """..."""
    if root:
        yield root.data

        if root.left:
            for node in _pre_order(root.left):
                yield node

        if root.right:
            for node in _pre_order(root.right):
                yield node

def _post_order(root):
    """..."""
    if root:
        if root.left:
            for node in _post_order(root.left):
                yield node

        if root.right:
            for node in _post_order(root.right):
                yield node

        yield root.data

def iter_tree(root, order="in"):
    if order == "in":
        return _in_order(root)
    elif order == "pre":
        return _pre_order(root)
    elif order == "post":
        return _post_order(root)
    else:
        raise ValueError("order must be one of [in, pre, or post]")

def bft(root):
    """Breadth first search"""
    queue = deque([root])
    out = deque([])
    while queue:
        current = queue.popleft()
        out.append(current.data)

        if current.left is not None:
            queue.append(current.left)

        if current.right is not None:
            queue.append(current.right)

    return out

def iter_bft(root, reverse=False):
    """contains queue/stack combo
    """
    # perform bread first search
    out = bft(root)
    while out:
        if reverse:
            # stack
            yield out.pop()
        else:
            # queue
            yield out.popleft()
