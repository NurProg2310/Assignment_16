from typing import List, Any, Dict, Set, Generator

class StaticArray:
    def __init__(self, capacity: int):
        """
        Initialize a static array of a given capacity.
        """
        self.capacity = capacity
        self.data = [None]*capacity

    def set(self, index: int, value: int) -> None:
        """
        Set the value at a particular index.
        """
        self.data[index] = value

    def get(self, index: int) -> int:
        """
        Retrieve the value at a particular index.
        """
        return  self.data[index]

class DynamicArray:
    def __init__(self):
        """
        Initialize an empty dynamic array.
        """
        self.data  = []
    def append(self, value: int) -> None:
        """
        Add a value to the end of the dynamic array.
        """
        self.data.append(value)

    def insert(self, index: int, value: int) -> None:
        """
        Insert a value at a particular index.
        """
        self.data.insert(index,value)

    def delete(self, index: int) -> None:
        """
        Delete the value at a particular index.
        """
        self.data.pop(index)

    def get(self, index: int) -> int:
        """
        Retrieve the value at a particular index.
        """
        return self.data[index]

class Node:
    def __init__(self, value: int):
        """
        Initialize a node.
        """
        self.value = value
        self.next = None

class SinglyLinkedList:
    def __init__(self):
        """
        Initialize an empty singly linked list.
        """
        self.head = None

    def append(self, value: int) -> None:
        """
        Add a node with a value to the end of the linked list.
        """
        new_node = Node(value)
        if self.head is None:
            self.head = new_node
            return
        current = self.head
        while current.next is not None:
            current = current.next
        current.next = new_node



    def insert(self, position: int, value: int) -> None:
        """
        Insert a node with a value at a particular position.
        """
        new_node = Node(value)
        if position ==0:
            new_node.next =self.head
            return
        current = self.head
        index = 0
        while current is not None and index < position-1:
            current = current.next
            index+=1

        if current is None:
            raise ValueError (f"Position out of range")
        new_node.next = current.next
        current.next = new_node




    def delete(self, value: int) -> None:
        if self.head is None:
            return
        if self.head.value == value:
            self.head = self.head.next
            return
        current = self.head
        while current.next is not None and current.next.value != value:
            current = current.next
        if current.next is not None:
            current.next = current.next.next

    def find(self, value: int):
        current = self.head
        while current is not None:
            if current.value == value:
                return current
            current = current.next
        return None

    def size(self) -> int:
        count = 0
        current = self.head
        while current is not None:
            count += 1
            current = current.next
        return count

    def is_empty(self) -> bool:
        return self.head is None

    def print_list(self) -> None:
        current = self.head
        result = []
        while current is not None:
            result.append(str(current.value))
            current = current.next
        print(" -> ".join(result))

    def reverse(self) -> None:
        prev = None
        current = self.head
        while current is not None:
            nxt = current.next
            current.next = prev
            prev = current
            current = nxt
        self.head = prev

    def get_head(self):
        return self.head

    def get_tail(self):
        current = self.head
        if current is None:
            return None
        while current.next is not None:
            current = current.next
        return current

class DoubleNode:
    def __init__(self, value: int, next_node = None, prev_node = None):
        """
        Initialize a double node with value, next, and previous.
        """
        self.value = value
        self.next = next_node
        self.prev = prev_node

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def append(self, value: int) -> None:
        new_node = DoubleNode(value)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
            return
        self.tail.next = new_node
        new_node.prev = self.tail
        self.tail = new_node

    def insert(self, position: int, value: int) -> None:
        new_node = DoubleNode(value)
        if position == 0:
            if self.head is None:
                self.head = new_node
                self.tail = new_node
                return
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
            return
        current = self.head
        index = 0
        while current is not None and index < position:
            current = current.next
            index += 1
        if current is None:
            self.append(value)
            return
        prev_node = current.prev
        prev_node.next = new_node
        new_node.prev = prev_node
        new_node.next = current
        current.prev = new_node

    def delete(self, value: int) -> None:
        current = self.head
        while current is not None and current.value != value:
            current = current.next
        if current is None:
            return
        if current.prev is None:
            self.head = current.next
            if self.head is not None:
                self.head.prev = None
            else:
                self.tail = None
            return
        if current.next is None:
            self.tail = current.prev
            self.tail.next = None
            return
        current.prev.next = current.next
        current.next.prev = current.prev

    def find(self, value: int):
        current = self.head
        while current is not None:
            if current.value == value:
                return current
            current = current.next
        return None

    def size(self) -> int:
        count = 0
        current = self.head
        while current is not None:
            count += 1
            current = current.next
        return count

    def is_empty(self) -> bool:
        return self.head is None

    def print_list(self) -> None:
        current = self.head
        values = []
        while current is not None:
            values.append(str(current.value))
            current = current.next
        print(" <-> ".join(values))

    def reverse(self) -> None:
        current = self.head
        self.tail = current
        prev = None
        while current is not None:
            nxt = current.next
            current.next = prev
            current.prev = nxt
            prev = current
            current = nxt
        self.head = prev

    def get_head(self):
        return self.head

    def get_tail(self):
        return self.tail

class Queue:
    def __init__(self):
        self.items = []

    def enqueue(self, value: int) -> None:
        self.items.append(value)

    def dequeue(self) -> int:
        if not self.items:
            return None
        return self.items.pop(0)

    def peek(self) -> int:
        if not self.items:
            return None
        return self.items[0]

    def is_empty(self) -> bool:
        return len(self.items) == 0

class TreeNode:
    def __init__(self, value: int):
        """
        Initialize a tree node with value.
        """
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        """
        Initialize an empty binary search tree.
        """
        self.root = None

    def insert(self, value: int) -> None:
        """
        Insert a node with a specific value into the binary search tree.
        """
        if self.root is None:
            self.root = TreeNode(value)
            return

        current = self.root
        while True:
            if value < current.value:
                if current.left is None:
                    current.left = TreeNode(value)
                    return
                current = current.left
            else:
                if current.right is None:
                    current.right = TreeNode(value)
                    return
                current = current.right

    def delete(self, value: int) -> None:
        """
        Remove a node with a specific value from the binary search tree.
        """
        self.root = self._delete_recursive(self.root, value)

    def _delete_recursive(self, node, value):
        if node is None:
            return None

        if value < node.value:
            node.left = self._delete_recursive(node.left, value)
        elif value > node.value:
            node.right = self._delete_recursive(node.right, value)
        else:
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left

            min_larger_node = node.right
            while min_larger_node.left:
                min_larger_node = min_larger_node.left

            node.value = min_larger_node.value
            node.right = self._delete_recursive(node.right, min_larger_node.value)

        return node

    def search(self, value: int) -> TreeNode:
        """
        Search for a node with a specific value in the binary search tree.
        """
        current = self.root
        while current is not None:
            if value == current.value:
                return current
            elif value < current.value:
                current = current.left
            else:
                current = current.right
        return None

    def inorder_traversal(self) -> List[int]:
        """
        Perform an in-order traversal of the binary search tree.
        """
        result = []

        def traverse(node):
            if node:
                traverse(node.left)
                result.append(node.value)
                traverse(node.right)

        traverse(self.root)
        return result

    def size(self) -> int:
        """
        Returns the number of nodes in the tree.
        """
        def count(node):
            if node is None:
                return 0
            return 1 + count(node.left) + count(node.right)

        return count(self.root)

    def is_empty(self) -> bool:
        """
        Checks if the tree is empty.
        """
        return self.root is None

    def height(self) -> int:
        """
        Returns the height of the tree.
        """
        def h(node):
            if node is None:
                return -1
            return 1 + max(h(node.left), h(node.right))

        return h(self.root)

    def preorder_traversal(self) -> List[int]:
        """
        Perform a pre-order traversal of the tree.
        """
        result = []

        def traverse(node):
            if node:
                result.append(node.value)
                traverse(node.left)
                traverse(node.right)

        traverse(self.root)
        return result

    def postorder_traversal(self) -> List[int]:
        """
        Perform a post-order traversal of the tree.
        """
        result = []

        def traverse(node):
            if node:
                traverse(node.left)
                traverse(node.right)
                result.append(node.value)

        traverse(self.root)
        return result

    def level_order_traversal(self) -> List[int]:
        """
        Perform a level order (breadth-first) traversal of the tree.
        """
        if self.root is None:
            return []

        result = []
        queue = [self.root]

        while queue:
            node = queue.pop(0)
            result.append(node.value)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        return result

    def minimum(self) -> TreeNode:
        """
        Returns the node with the minimum value in the tree.
        """
        current = self.root
        if current is None:
            return None
        while current.left:
            current = current.left
        return current

    def maximum(self) -> TreeNode:
        """
        Returns the node with the maximum value in the tree.
        """
        current = self.root
        if current is None:
            return None
        while current.right:
            current = current.right
        return current

    def is_valid_bst(self) -> bool:
        """
        Check if the tree is a valid binary search tree.
        """
        def validate(node, low, high):
            if node is None:
                return True
            if node.value <= low or node.value >= high:
                return False
            return validate(node.left, low, node.value) and validate(node.right, node.value, high)

        return validate(self.root, float("-inf"), float("inf"))

def insertion_sort(lst: List[int]) -> List[int]:
    arr = lst.copy()
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def selection_sort(lst: List[int]) -> List[int]:
    arr = lst.copy()
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


def bubble_sort(lst: List[int]) -> List[int]:
    arr = lst.copy()
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def shell_sort(lst: List[int]) -> List[int]:
    arr = lst.copy()
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2
    return arr


def merge_sort(lst: List[int]) -> List[int]:
    if len(lst) <= 1:
        return lst
    mid = len(lst) // 2
    left = merge_sort(lst[:mid])
    right = merge_sort(lst[mid:])
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def quick_sort(lst: List[int]) -> List[int]:
    if len(lst) <= 1:
        return lst
    pivot = lst[len(lst) // 2]
    left = [x for x in lst if x < pivot]
    mid = [x for x in lst if x == pivot]
    right = [x for x in lst if x > pivot]
    return quick_sort(left) + mid + quick_sort(right)