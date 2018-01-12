# Authors: Joseph Knox josephk@allenistitute.org
# License: 

from collections import deque

class Node(object):
    """..."""
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

def in_order(self, node, result=[]):
    """..."""
    if node:
        in_order(node.left, result=result)
        result.append(node.data)
        in_order(node.right, result=result)
  
    return result

def pre_order(node, result=[]):
    """..."""
    if node:
        result.append(node.data)
        pre_order(node.left, result=result)
        pre_order(node.right, result=result)
    
    return result

def post_order(node, result=[]):
    """..."""
    if node:
        post_order(node.left, base=base, result=result)
        post_order(node.right, base=base, result=result)
        result.append(node.data)
    
    return result

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
