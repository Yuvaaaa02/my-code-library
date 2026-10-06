# My Code Library (Cloud Code Storage)

This repository serves as the remote cloud storage backend for the **MyCodes** Python package.

## Repository Structure

```text
my-code-library/
│
├── codes/
│   ├── 001.py
│   ├── 002.py
│   ├── 003.py
│   ├── 101.py
│   ├── 102.py
│   └── ...
│
├── index.json
└── README.md
```

## How to Add New Code Snippets

1. **Create the code file** under `codes/<id>.py` (e.g. `codes/103.py`):
   ```python
   def quick_sort(arr):
       if len(arr) <= 1:
           return arr
       pivot = arr[len(arr) // 2]
       left = [x for x in arr if x < pivot]
       middle = [x for x in arr if x == pivot]
       right = [x for x in arr if x > pivot]
       return quick_sort(left) + middle + quick_sort(right)
   ```

2. **Add metadata to `index.json`**:
   ```json
   "103": {
     "name": "Quick Sort",
     "description": "Quicksort algorithm implementation",
     "category": "DSA",
     "language": "python",
     "tags": ["sort", "quicksort", "dsa", "algorithms"],
     "file": "codes/103.py"
   }
   ```

3. **Commit and push** to your GitHub repository:
   ```bash
   git add codes/103.py index.json
   git commit -m "Add code 103: Quick Sort"
   git push origin main
   ```

4. **Instantly access on any computer**:
   ```python
   from mycodes import get_code
   code = get_code(103)
   ```
   Or inside Jupyter Notebook:
   ```python
   %load_code 103
   ```
