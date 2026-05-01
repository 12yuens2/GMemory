LCB_CODEGEN_SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given a problem specification "
    "and will generate a correct Python program that matches the specification and passes all tests. "
    "Enclose your solution within triple-backtick Python code fences."
)

LCB_CODEEXEC_SYSTEM_PROMPT = (
    "You are an expert at Python programming and code execution. "
    "Given a Python function and an input to that function, you will determine the exact output "
    "when the function is executed. Provide only the assertion with the filled-in output inside "
    "[ANSWER] and [/ANSWER] tags."
)

LCB_TESTPRED_SYSTEM_PROMPT = (
    "You are an expert at Python programming and test analysis. "
    "Given a problem description and a starter function, you will predict the expected output "
    "for a specific test case. Provide the full assertion with the correct output inside "
    "[ANSWER] and [/ANSWER] tags."
)

LCB_SELFREPAIR_SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given a problem specification, a failing "
    "solution, and the error it produced. Identify the issue and provide a corrected solution. "
    "Enclose your fixed solution within triple-backtick Python code fences."
)


# ---------------------------------------------------------------------------
# Static few-shot examples
# ---------------------------------------------------------------------------

LCB_CODEGEN_FEW_SHOTS = [
    '''Problem: Given an array of integers nums and an integer target, return the indices of the two numbers that add up to target. You may assume exactly one solution exists and you may not use the same element twice.

Example:
Input: nums = [2, 7, 11, 15], target = 9
Output: [0, 1]

Solution:
```python
from typing import List

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in seen:
                return [seen[complement], i]
            seen[num] = i
        return []
```''',
]

LCB_CODEEXEC_FEW_SHOTS = [
    """Function:
def running_total(nums):
    result = []
    total = 0
    for n in nums:
        total += n
        result.append(total)
    return result

Input:
assert running_total([1, 2, 3, 4]) == ??

Trace: total starts at 0; after 1 → [1]; after 2 → [1,3]; after 3 → [1,3,6]; after 4 → [1,3,6,10].

[ANSWER]assert running_total([1, 2, 3, 4]) == [1, 3, 6, 10][/ANSWER]""",
]

LCB_TESTPRED_FEW_SHOTS = [
    """Problem: Given a list of strings, return a new list containing only the strings that have more than 3 characters.

Starter code:
def filter_long(words: list[str]) -> list[str]:
    pass

Test input:
filter_long(["hi", "hello", "yes", "world", "no"])

Reasoning: The problem asks for strings with more than 3 characters. "hi" (2) and "no" (2) are excluded; "yes" (3) is excluded (not strictly more than 3); "hello" (5) and "world" (5) are included.

[ANSWER]assert filter_long(["hi", "hello", "yes", "world", "no"]) == ["hello", "world"][/ANSWER]""",
]

LCB_SELFREPAIR_FEW_SHOTS = [
    '''Problem: Return the n-th Fibonacci number (0-indexed: fib(0) = 0, fib(1) = 1, fib(2) = 1, ...).

Failing solution:
```python
def fib(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a
```

Error: AssertionError: fib(5) returned 3 instead of expected 5.

Analysis: The loop correctly runs n-1 times and swaps (a, b) each iteration, so after the loop b holds fib(n). Returning a instead of b is the bug.

Fixed solution:
```python
def fib(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b
```''',
]
