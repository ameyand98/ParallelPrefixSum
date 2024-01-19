import argparse
import random

parser = argparse.ArgumentParser()

parser.add_argument('--num_inputs', type=int, required=True)
parser.add_argument('--file_name', type=str, required=True)
parser.add_argument('--type', type=str, required=False, default='int')
parser.add_argument('--dims', type=int, required=False, default=0)
args = parser.parse_args()

with open(args.file_name, 'w') as f:
  f.write('0' if args.type == 'int' else str(args.dims))
  f.write('\n')
  f.write(str(args.num_inputs))
  f.write('\n')
  for i in range(args.num_inputs):
    if args.type == 'int': 
	  f.write(str(random.randint(0, 10000)))
	  f.write('\n')
    else:
	  for j in range(args.dims-1):
		f.write(str(random.random()))
		f.write(', ')
	  f.write(str(random.random()))
	  f.write('\n')
	  