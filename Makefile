default:
	@g++ -O3 -o pfxsum pfxsum.cc barrier.cc -lpthread -Wall

clean:
	rm -rf pfxsum *.o *.d *.out