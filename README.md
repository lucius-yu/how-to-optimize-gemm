# How To Optimize Gemm wiki pages

GEMM stands for General Matrix Multiplication. I cloned the repo https://github.com/flame/how-to-optimize-gemm/wiki

Copyright by Prof. Robert van de Geijn (rvdg@cs.utexas.edu).

Adapation by Lucius Yu for running this demo easily. 

* I use cmake to generate makefile for building
* I change the output format to csv. which will be easy to handled with python code
* I write simple python script load_data.py to explore the performance


## Build project

```
cd build
cmake ../src -DNEW=MMult1 -DOUT=../data/MMult1; make clean; make; make test
```

* In example, the MMult1.c will be compiled 
* In example, the result will be saved into ../data/MMult1.csv
* All cmake and make generated files will be in build directory. You can clean all of them
* NEW need to be set as correct C source code which you implement matrix operation
* It is better always to do clean before build and test

## how to clean cmake generated files

Use git to clean files which is not under track. 'git clean -d -f -x -n'. Note: -n means dry-run

