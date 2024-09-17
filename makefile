CC=g++
CFLAGS= -O3

FILES=./src/main.cpp

all:
	$(CC) $(CFLAGS) $(FILES)

.PHONY: clean

clean:
	rm a.out