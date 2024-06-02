from collections import OrderedDict
from dataclasses import dataclass
from typing import Self, Any

# @dataclass
# class ListNode:
#     val: int = None
#     left: Self = None
#     right: Self = None

# class lruCache:

#     def __init__(self, experts) -> None:
#         self.e_to_node = OrderedDict()
#         self.head, self.tail = ListNode(), ListNode()
#         self.build_doubly_linked_list(experts)

#     def build_doubly_linked_list(self, experts):
#         l = self.head
#         for e in experts:
#             l = self.e_to_node[e] = ListNode(val=e, left=l)
#         self.tail.left = l

#         r = self.tail
#         for node in reversed(self.e_to_node.values()):
#             node.right = r
#             r = node
#         self.head.right = r

#     def make_last(self, node):
#         # sew neighbors together
#         node.left.right = node.right
#         node.right.left = node.left

#         org_last = self.tail.left
#         org_last.right = node
#         self.tail.left = node
#         node.left = org_last
#         node.right = self.tail

#     def get_lru(self) -> int:
#         lru = self.head.right
#         self.make_last(lru)
#         return lru.val

#     def use(self, e) -> None:
#         node = self.e_to_node[e]
#         self.make_last(node)

class lruCache(OrderedDict):
    # inspired by:
    # https://docs.python.org/3/library/collections.html#collections.OrderedDict
    # https://stackoverflow.com/questions/21062781/shortest-way-to-get-first-item-of-ordereddict-in-python-3

    def get_lru(self) -> Any:
        k = next(iter(self))
        self.move_to_end(k)
        return k

# def print_lru_cache(lru_cache):
#     ptr = lru_cache.head
#     while ptr is not None:
#         print(ptr.val)
#         ptr = ptr.right

# def test0():
#     experts = [0, 1, 2, 3]
#     lru_cache = lruCache(experts)
#     print_lru_cache(lru_cache)

# def test1():
#     experts = [0, 1, 2, 3]
#     lru_cache = lruCache(experts)
#     lru_cache.use(1)
#     lru_cache.use(2)
#     lru_cache.get_lru()
#     print_lru_cache(lru_cache)

# def test2():
#     experts = [0, 1, 2, 3]
#     lru_cache = lruCache(experts)
#     lru_cache.use(0)
#     lru_cache.use(0)
#     lru_cache.get_lru()
#     lru_cache.use(1)
#     print_lru_cache(lru_cache)

def test3():
    experts = [0, 1, 2, 3]
    lru_cache = lruCache.fromkeys(experts)
    lru_cache.move_to_end(1)
    lru_cache.move_to_end(2)
    print(f"lru: {lru_cache.get_lru()}")
    print(lru_cache)

def test4():
    experts = [0, 1, 2, 3]
    lru_cache = lruCache.fromkeys(experts)
    lru_cache.move_to_end(0)
    lru_cache.move_to_end(0)
    lru_cache.get_lru()
    lru_cache.move_to_end(1)
    print(f"lru: {lru_cache.get_lru()}")
    print(lru_cache)

if __name__ == "__main__":
    # test0()
    # test1()
    # test2()
    # test3()
    test4()
