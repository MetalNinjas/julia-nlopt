.PHONY: all

all: nlopt_wrapper.so

nlopt_wrapper.so: nlopt_wrapper.c
	g++ -xc -shared $< -o $@ -fPIC -lnlopt_cxx
