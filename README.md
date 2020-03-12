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

