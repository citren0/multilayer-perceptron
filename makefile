CC=g++
CFLAGS= -O3
FILES=./src/main.cpp ./src/Perceptron.cpp ./src/Helpers.cpp

all:
	$(CC) $(CFLAGS) $(FILES)

.PHONY: clean

clean:
	rm a.out