import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--foo', help='foo help',type=int)

args = parser.parse_args()

arg_vars=vars(args)

print(arg_vars)