# PyZX - Python library for quantum circuit rewriting 
#        and optimization using the ZX-calculus
# Copyright (C) 2018 - Aleks Kissinger and John van de Wetering

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from . import architecture

debug = False

def steiner_gauss(matrix, architecture, full_reduce=False, x=None, y=None):
    """
    Performs Gaussian elimination that is constraint by the given architecture
    
    :param matrix: PyZX Mat2 matrix to be reduced
    :param architecture: The Architecture object to conform to
    :param full_reduce: Whether to fully reduce or only create an upper triangular form
    :param x: 
    :param y: 
    :return: Rank of the given matrix
    """
    def row_add(c0, c1):
        matrix.row_add(c0, c1)
        debug and print("Reducing", c0, c1)
        if x != None: x.row_add(c0, c1)
        if y != None: y.col_add(c1, c0)
    def steiner_reduce(col, root, nodes, upper):
        steiner_tree = architecture.steiner_tree(root, nodes, upper)
        # Remove all zeros
        next_check = next(steiner_tree)
        debug and print("Step 1: remove zeros")
        if upper:
            zeros = []
            while next_check is not None:
                s0, s1 = next_check
                if matrix[s0, col] == 0:  # s1 is a new steiner point or root = 0
                    zeros.append(next_check)
                next_check = next(steiner_tree)
            while len(zeros) > 0:
                s0, s1 = zeros.pop(-1)
                if matrix[s0, col] == 0:
                    row_add(s1, s0)
                    debug and print(matrix[s0, col], matrix[s1, col])
        else:
            debug and print("deal with zero root")
            if next_check is not None and matrix[next_check[0], col] == 0:  # root is zero
                print("WARNING : Root is 0 => reducing non-pivot column", matrix.data)
            debug and print("Step 1: remove zeros", matrix[:, col])
            while next_check is not None:
                s0, s1 = next_check
                if matrix[s1, col] == 0:  # s1 is a new steiner point
                    row_add(s0, s1)
                next_check = next(steiner_tree)
        # Reduce stuff
        debug and print("Step 2: remove ones")
        next_add = next(steiner_tree)
        while next_add is not None:
            s0, s1 = next_add
            row_add(s0, s1)
            next_add = next(steiner_tree)
            debug and print(next_add)
        debug and print("Step 3: profit")

    rows = matrix.rows()
    cols = matrix.cols()
    p_cols = []
    pivot = 0
    current_row = 0

    for current_row in range(rows):
        found_pivot = False
        while not found_pivot and pivot < cols:
            nodes = [r for r in range(current_row,rows)
                       if matrix[r,pivot] == 1]
            if len(nodes) > 0:
                p_cols.append(pivot)
                found_pivot = True
            else:
                pivot += 1
        # no more pivots left
        if not found_pivot: break

        nodes.insert(0,current_row)
        steiner_reduce(pivot, current_row, nodes, True)
        pivot+=1



    # for c in range(cols):
    #     nodes = [r for r in range(pivot, rows) if pivot==r or matrix[r,c] == 1]
    #     steiner_reduce(c, pivot, nodes, True)
    #     if matrix[pivot,c] == 1:
    #         p_cols.append(c)
    #         pivot += 1

    debug and print("Upper triangle form", matrix)
    rank = len(p_cols)
    debug and print(p_cols)

    if full_reduce:
        for current_row in reversed(range(len(p_cols))):
            pivot = p_cols[current_row]
            nodes = [r for r in range(0, current_row) if matrix[r,pivot] == 1]
            if len(nodes) > 0:
                nodes.append(current_row)
                steiner_reduce(pivot, current_row, nodes, False)

    # if full_reduce:
    #     pivot -= 1
    #     for c in reversed(p_cols):
    #         debug and print(pivot, matrix[:,c])
    #         nodes = [r for r in range(0, pivot+1) if r==pivot or matrix[r,c] == 1]
    #         if len(nodes) > 1:
    #             steiner_reduce(c, pivot, nodes, False)
    #         pivot -= 1
    return rank
