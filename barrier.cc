#include "barrier.h"


int my_barrier_init(my_barrier* barrier, unsigned int count) {
    if (count == 0) {
        return -1;
    }

    int ret;
    ret = pthread_mutex_init(&barrier->mutex, NULL);
    if (ret) return ret;

    ret = pthread_cond_init(&barrier->cond, NULL);
    if (ret) return ret;

    barrier->count = count;
    barrier->left = count;
    barrier->round = 0;
    return 0;
}

int my_barrier_destroy(my_barrier* barrier) {
    if (barrier->count == 0) {
        return -1;
    }
    
    barrier->count = 0;
    pthread_mutex_destroy(&barrier->mutex);
    pthread_cond_destroy(&barrier->cond);
    return 0;
}

int my_barrier_wait(my_barrier* barrier) {
    pthread_mutex_lock(&barrier->mutex);

    if (--barrier->left) {
        unsigned round = barrier->round;
		do {
			pthread_cond_wait(&barrier->cond, &barrier->mutex);
		} while (round == barrier->round);
		pthread_mutex_unlock(&barrier->mutex);
    } else {
        barrier->round += 1;
		barrier->left = barrier->count;
		pthread_cond_broadcast(&barrier->cond);
		pthread_mutex_unlock(&barrier->mutex);
    }

    return 0;
}