#include<time.h>
#include<stdio.h>
#include<iostream>



double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(){

	get_time();

	return 0;

}
