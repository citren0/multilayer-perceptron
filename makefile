CC=g++
CFLAGS= -O3 -fopenmp
LIBS = 
FILES=./src/main.cpp ./src/Perceptron.cpp ./src/Helpers.cpp

all:
	$(CC) $(CFLAGS) $(FILES) $(LIBS)

.PHONY: clean

clean:
	rm a.out