.PHONY: v1 v2 v3

v1: sumArrayOnHost.c
	nvcc -Xcompiler -std=c99 $< -o $@
	./$@

v2: sumArrays.cu
	nvcc -arch sm_60 $< -o $@
	./$@

v3: deviceData.cu
	nvcc -arch sm_60 $< -o $@
	./$@
