import argparse

def main(args):
    return f"{args.operation}: {args.a + args.b}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", default= 'add', type= str)
    parser.add_argument("--a", default= 0, type= int)
    parser.add_argument("--b", default= 0, type= int)
    args = parser.parse_args()
    print(main(args))