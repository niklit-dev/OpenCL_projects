CXX=g++

CXXFLAGS = -c  
CXXFLAGS += -std=c++14



LIBS= -lOpenCL

INCLUDE=-I /usr/include/CL

all: main

main: main.o 
	$(CXX) main.o -o main $(LIBS)

main.o:
	$(CXX) $(CXXFLAGS) $(INCLUDE) main.cpp 

clean:
	rm -rf main *.o
