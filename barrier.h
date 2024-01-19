#include <pthread.h>
#include <atomic>

typedef struct {
	pthread_mutex_t mutex;
	pthread_cond_t cond;
	unsigned count;
	unsigned left;
	unsigned round;
} my_barrier;


extern int my_barrier_init(my_barrier* barrier, unsigned int count);

extern int my_barrier_destroy(my_barrier* barrier);

extern int my_barrier_wait(my_barrier* barrier);