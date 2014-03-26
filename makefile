all: 
	g++ -c -fopenmp pLDA.cpp GSsdk.h GSsdk.cpp -std=c++0x
	g++ -o fastLDA -fopenmp pLDA.cpp GSsdk.h GSsdk.cpp -std=c++0x
run:
	./fastLDA docword.nytimes.txt 50 100
clean:
	rm -f fastLDA.o fastLDA