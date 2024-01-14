GCC=/usr/bin/gcc
GCCFLAGS=-Wall -pedantic -std=c11 -fms-extensions -O2 -flto

bin/perfs: perfs.c util.c util.h
	$(GCC) $(GCCFLAGS) util.c perfs.c -o bin/perfs

clean:
	rm -f bin/perfs
