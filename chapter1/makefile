.PHONY: v1 v2

v1: hello.cu
	nvcc hello.cu -o v1
	./v1

v2: hello.cu
	nvcc -arch sm_50 hello.cu -o v2
	./v2
