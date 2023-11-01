# How To Optimize Gemm wiki pages

https://github.com/flame/how-to-optimize-gemm/wiki

Copyright by Prof. Robert van de Geijn (rvdg@cs.utexas.edu).


Adaption by Lucius Yu

* Using CMake to build whole project
* Using python to do plot and analysis


## Build project

```
cd build
cmake ../src -DNEW=MMult1 -DOUT=../data/MMult1
make
make test
```

* In example, the MMult1.c will be compiled 
* In example, the result will be saved into ../data/MMult1.csv
* All cmake and make generated files will be in build directory. You can clean all of them
* NEW_SRC need to be set as correct C source code which you implement matrix operation

## how to clean cmake generated files

Use git to clean files which is not under track. 'git clean -d -f -x -n'. Note: -n means dry-run

