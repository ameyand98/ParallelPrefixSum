#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <pthread.h>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <string>
#include "barrier.h"
using namespace std;

pthread_barrier_t barrier;
my_barrier bar;
static bool is_float = false;
static bool my_bar = false;

template <typename T>
struct thread_args {
  int id;
  int num_threads;
  char* in_name;
  char* out_name;
  void* array;
  int* num_arrived;
  int block_size;
  int org_size;
  int dim;
};

void* seq_parse_file(char* name, int* org_size);
void par_parse_file(char* name, int thread_id, void* array, int block_size, int org_size, int dim);
int parse_args(char** argv, char** in_name, char** out_name, int* num_thr);
int write_file(void* array, char* file, int org_size);

template <typename T>
T scan_op(T x, T y);

template <typename T>
int seq_pfxscan(char* in_name, char* out_name, int* org_size);

template <typename T>
int par_pfxsum(char* in_name, char* out_name, int num_threads, int* org_size);

template <typename T>
int per_thread_scan(thread_args<T>* args);

template <typename T>
int scan(char* in_name, char* out_name, int num_threads);

int pad_array_size(unsigned int act_size);
bool is_power_of_two(int n);

template <typename T>
T scan_op(T x, T y);
int add_element(int x, int y);
vector<float>* add_element(vector<float>* x, vector<float>* y);
vector<float>* create_empty(int size);
int* get_dim(char* name);

template <typename T>
T create_deep_cpy(T array);
vector<float>* deep_cpy(vector<float>* array);
int deep_cpy(int val);


// start here
int main(int argc, char** argv) {
  // arg parsing
  char* in_name;
  char* out_name;
  int num_threads;

  int ret_val = parse_args(argv, &in_name, &out_name, &num_threads);
  if (ret_val) return ret_val;
  int* vals = get_dim(in_name);
  int dim = vals[0];

  if (dim) {
    scan<vector<float>*>(in_name, out_name, num_threads);
  } else {
    scan<int>(in_name, out_name, num_threads);
  }

  return 0;
}


// main generic scan function
template <typename T>
int scan(char* in_name, char* out_name, int num_threads) {
  int* org_size = (int*) malloc(sizeof(int));
  auto start = chrono::steady_clock::now();

  if (num_threads) {
    par_pfxsum<T>(in_name, out_name, num_threads, org_size);
  } else {
    seq_pfxscan<T>(in_name, out_name, org_size);
  }

  auto elapsed = chrono::steady_clock::now() - start;
  auto seconds = chrono::duration_cast<chrono::milliseconds>(elapsed).count();
  cout.precision(6);
  cout << (float) seconds / 1000 << endl;
  return 0;
}

// sequential prefix scan
template <typename T>
int seq_pfxscan(char* in_name, char* out_name, int* org_size) {
  vector<T>* in_arr = (vector<T>*) seq_parse_file(in_name, org_size);

  int size = in_arr->size();
  for (int i = 1; i < size; i++) {
    in_arr->at(i) = scan_op<T>(in_arr->at(i), in_arr->at(i - 1)); 
  }

  write_file(in_arr, out_name, *org_size);
  return 0;
}


// create a deep cpy of the array
vector<float>* deep_cpy(vector<float>* array) {
  int size = array->size();
  vector<float>* new_arr = new vector<float>(size);
  for (int i = 0; i < size; i++) {
    new_arr->at(i) = array->at(i);
  }
  return new_arr;
}

int deep_cpy(int val) {
  return val;
}

template <typename T>
T create_deep_cpy(T array) {
  return deep_cpy(array);
}


// parallel prefix sum functions
template <typename T>
int par_pfxsum(char* in_name, char* out_name, int num_threads, int* org_size) {
  if (my_bar) {
    my_barrier_init(&bar, num_threads);
  } else {
    pthread_barrier_init(&barrier, NULL, (unsigned int) num_threads);
  }

  vector<vector<float>*>* float_arr = 0;
  vector<int>* int_arr = 0;
  int* vals = get_dim(in_name);
  int dim = vals[0];
  int size = vals[1];
  *org_size = size;
  int actual_size = size;

  if (!is_power_of_two(size)) actual_size = pad_array_size(size);

  if (dim > 0) is_float = true;
  if (is_float) {
    float_arr = new vector<vector<float>*>(actual_size);  
  } else {
    int_arr = new vector<int>(actual_size);
  }

  int block_size = actual_size / num_threads;
  if (actual_size % num_threads != 0) block_size++;

  
  pthread_t ids[num_threads];
  int num_arrived = 0;

  for (int i = 0; i < num_threads; i++) {
    thread_args<T>* args = (thread_args<T>*) malloc(sizeof(thread_args<T>));
    args->id = i;
    // args->array = is_float ? float_arr : int_arr;
    if (is_float) {
      args->array = (void*) float_arr;
    } else {
      args->array = (void*) int_arr;
    }
    args->num_threads = num_threads;
    // args->arr_size = arr_size;
    args->num_arrived = &num_arrived;
    args->in_name = in_name;
    args->out_name = out_name;
    args->block_size = block_size;
    args->org_size = *org_size;
    args->dim = dim;
    // cout << "made args for thread" << endl;
    pthread_create(&ids[i], NULL, (void* (*)(void*)) per_thread_scan<T>, args);
  }

  for (int i = 0; i < num_threads; i++) {
    // cout << "----ending thread" << endl;
    pthread_join(ids[i], NULL);
  }

  write_file(is_float ? (void*) float_arr : (void*) int_arr, out_name, *org_size);

  if (my_bar) {
    my_barrier_destroy(&bar);
  } else {
    pthread_barrier_destroy(&barrier);
  }

  return 0;
}


// per thread function for the parallel prefix scan 
template <typename T>
int per_thread_scan(thread_args<T>* args) {
  int thread_id = args->id;
  int num_threads = args->num_threads;

  // parallel parsing of file
  par_parse_file(args->in_name, thread_id, (void*) args->array, args->block_size, args->org_size, args->dim);

  if (my_bar) {
      my_barrier_wait(&bar);
    } else {
      pthread_barrier_wait(&barrier);
  }

  vector<T>* array = (vector<T>*) args->array;
  int num_elements = array->size();
  int offset = 1;

  int mult_one; int mult_two;

  // up-sweep reduction step
  for (int num_ops = num_elements >> 1; num_ops > 0; num_ops >>= 1) {
    if (my_bar) {
      my_barrier_wait(&bar);
    } else {
      pthread_barrier_wait(&barrier);
    }

    int ops_per_thread = num_ops / num_threads;
    if (num_ops % num_threads != 0) ops_per_thread++;

    mult_one = 2 * thread_id * ops_per_thread + 1;
    mult_two = 2 * thread_id * ops_per_thread + 2;

    int index1 = offset * mult_one - 1;
    int index2 = offset * mult_two - 1;
    if (thread_id < num_ops) { // need to change this for load balancing
      for (int i = 0; i < ops_per_thread; i++) {
        if (index1 < num_elements && index2 < num_elements) {
          array->at(index2) = scan_op<T>(array->at(index2), array->at(index1));
          index1 += offset * 2;
          index2 += offset * 2;
        } else {
          break;
        }
      } 
    }
    
    offset *= 2;
  }


  if (thread_id == 0) { array->at(num_elements - 1) = create_deep_cpy<T>(array->at(0)); }

  // down-sweep step
  for (int num_ops = 1; num_ops < num_elements; num_ops *= 2) {
    offset >>= 1;
    if (my_bar) {
      my_barrier_wait(&bar);
    } else {
      pthread_barrier_wait(&barrier);
    }

    int ops_per_thread = num_ops / num_threads;
    if (num_ops % num_threads != 0) ops_per_thread++;

    mult_one = 2 * thread_id * ops_per_thread + 1;
    mult_two = 2 * thread_id * ops_per_thread + 2;

    int index1 = offset * mult_one - 1;
    int index2 = offset * mult_two - 1;
    if (thread_id < num_ops) { // need to update for load balancing 
      for (int i = 0; i < ops_per_thread; i++) {
        if (index1 < num_elements && index2 < num_elements) {
          T temp = array->at(index1);
          array->at(index1) = array->at(index2);
          array->at(index2) = scan_op<T>(temp, array->at(index1));
          index1 += offset * 2;
          index2 += offset * 2;
        } else {
          break;
        }
      }
    }
  }
  return 0;

}

int add_element(int x, int y) {
  return x + y;
}

vector<float>* add_element(vector<float>* x, vector<float>* y) {
  for (unsigned int i = 0; i < x->size(); i++) {
    x->at(i) += y->at(i);
  }
  return x;
}

template <typename T>
T scan_op(T x, T y) {
  return add_element(x, y);
}

// get dimension of float array
int* get_dim(char* name) {
  ifstream in_file(name);
  int dim; int size;
  in_file >> dim;
  in_file >> size;

  int* vals = (int*) malloc(sizeof(int) * 2);
  vals[0] = dim;
  vals[1] = size;

  in_file.close();
  return vals;
}

// paralle parse file function (per thread)
void par_parse_file(char* name, int thread_id, void* array, int block_size, int org_size, int dim) {
  ifstream in_file(name);

  if (is_float) {
    vector<vector<float>*>* arr = (vector<vector<float>*>*) array;
    unsigned int actual_size = arr->size();
    unsigned int start = thread_id * block_size;
    float val; char comma;
    if (start < actual_size) {
      in_file >> val;
      in_file >> val;
      for (unsigned int i = 0; i < start; i++) {
        for (int j = 0; j < dim; j++) {
          in_file >> val;
          if (j < dim - 1) in_file >> comma;
        }
      }
    } else { return; }

    for (int i = 0; i < block_size; i++) {
      if (start + i < actual_size) {
        if (start + i >= (unsigned int) org_size) {
          arr->at(start + i) = create_empty(dim);
        } else {
          vector<float>* new_arr = new vector<float>(dim);
          for (int j = 0; j < dim; j++) {
            in_file >> val;
            new_arr->at(j) = val;
            if (j < dim - 1) in_file >> comma;
          }
          arr->at(start + i) = new_arr;
        }
      } else { break; }
    }

    in_file.close();

  } else {
    vector<int>* arr = (vector<int>*) array;
    unsigned int actual_size = arr->size();
    unsigned int start = thread_id * block_size;
    int val;
    if (start < actual_size) {
      for (unsigned int i = 0; i < start + 2; i++) {
        in_file >> val;
      }
    } else { return; }

    for (int i = 0; i < block_size; i++) {
      if (start + i < actual_size) {
        if (start + i >= (unsigned int) org_size) {
          arr->at(start + i) = 0;
        } else {
          in_file >> val;
          arr->at(start + i) = val;
        }
      } else { break; }
    }
    in_file.close();

  }

}

// sequential parse file function
void* seq_parse_file(char* name, int* org_size) {
  ifstream in_file(name);

  int dim; int size;
  in_file >> dim;
  in_file >> size;
  *org_size = size;
  vector<vector<float>*>* float_arr = 0;
  vector<int>* int_arr = 0;
  int actual_size = size;
  // cout << (is_power_of_two(size)) << endl;

  if (!is_power_of_two(size)) actual_size = pad_array_size(size);

  // int* array;
  if (dim > 0) is_float = true;
  if (is_float) {
    float_arr = new vector<vector<float>*>(actual_size);  
  } else {
    int_arr = new vector<int>(actual_size);
  }

  // might be inefficient
  for (int i = 0; i < actual_size; i++) {
    if (!is_float) {
      if (i >= size) {
        int_arr->at(i) = 0;
      } else {
        int val;
        in_file >> val;
        int_arr->at(i) = val; 
      }
    } else {
      if (i >= size) {
        float_arr->at(i) = create_empty(dim);
      } else {
        char comma;
        vector<float>* new_arr = new vector<float>(dim);
        for (int j = 0; j < dim; j++) {
          float val;
          in_file >> val;
          new_arr->at(j) = val;
          // cout << new_arr->at(j);
          if (j < dim - 1) in_file >> comma;
          // cout << " ";
        }
        float_arr->at(i) = new_arr;
        // cout << "\n";

      }

    }
  } 

  in_file.close();

  return is_float ? (void*) float_arr : (void*) int_arr;
}

// create an array of zeroes 
vector<float>* create_empty(int size) {
  vector<float>* empty = new vector<float>(size);
  for (int i = 0; i < size; i++) {
    empty->at(i) = 0.000000000000;
    // cout << empty->at(i) << endl;
  }
  return empty;
}


// write output to the file  
int write_file(void* array, char* file, int org_size) {
  ofstream out_file(file);
  vector<vector<float>*>* float_arr = 0;
  vector<int>* int_arr = 0;

  if (is_float) {
    float_arr = (vector<vector<float>*>*) array;
  } else {
    int_arr = (vector<int>*) array;
  }

  for (int i = 0; i < org_size; i++) {
    if (!is_float) {
      int val = int_arr->at(i);
      out_file << val << "\n";
    } else {
      vector<float>* elem_arr = float_arr->at(i);
      int size = elem_arr->size();
      for (int j = 0; j < size; j++) {
        float val = elem_arr->at(j);
        out_file << val;
        if (j < size - 1) out_file << ", ";
      }
      out_file << "\n";
    }
    
  }
  out_file.close();
  
  return 0;

}

// trivial function to parse args in order
int parse_args(char** argv, char** in_name, char** out_name, int* num_thr) {
  string thread_par = (string) argv[1];
  if (thread_par.compare("-n") != 0) return -1;
  int num_threads = strtol(argv[2], NULL, 0);
  *num_thr = num_threads;

  string inp_par = (string) argv[3];
  if (inp_par.compare("-i") != 0) return -1;
  char* inp_file = argv[4];
  *in_name = inp_file;

  string out_par = (string) argv[5];
  if (out_par.compare("-o") != 0) return -1;
  char* out_file = argv[6];
  *out_name = out_file;

  char* bar_par = argv[7];
  // cout << bar_par[0] << endl;
  if (bar_par != 0) {
    if (bar_par[0] == '-' && bar_par[1] == 's') {
      my_bar = true;
    }
  }

  return 0;


}

// pad the array to the next power of two
int pad_array_size(unsigned int act_size){
  if (is_power_of_two(act_size)) return act_size;

  unsigned int total_size = act_size; 
  total_size--;
  total_size |= total_size >> 1;
  total_size |= total_size >> 2;
  total_size |= total_size >> 4;
  total_size |= total_size >> 8;
  total_size |= total_size >> 16;
  total_size++;

  return total_size; 
  
}


bool is_power_of_two(int n) {
   if (n == 0) return false;
   return (ceil(log2(n)) == floor(log2(n)));
}