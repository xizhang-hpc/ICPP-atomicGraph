# ICPP-atomicGraph
source code and data for comparing performance between atomic operations and graph coloring

1 How to compile
The compiling script is in file makeCUDA for both double and single precision on GPU Tesla V100 and K80 respectively.

2 Data
Mesh connectivity information including only Mesh1 and Mesh2 are supplied in folder data, due to storage limit in Github.
If more mesh information is needed, please contact Xi Zhang by email address zhangx299@mail.sysu.edu.cn

3 How to simulate
Put executable file into Mesh information folder, then execute. e.g.
./makeCUDA
tar -xzvf Mesh1.tgz
cp compareDoubleBest Mesh1
./compareDoubleBest

4 Algorithms and function No.
Function No: 1 for Algorithm 6
Function No: 2 for Algorithm 4
Function No: 3 for Algorithm 8
Function No: 4 for Algorithm 12
Function No: 5 for Algorithm 10
Function No: 6 for Algorithm 3
Function No: 7 for Algorithm 5
Function No: 8 for Algorithm 7
Function No: 9 for Algorithm 13
Function No: 10 for Algorithm 11
Function No: 11 for Algorithm 9

