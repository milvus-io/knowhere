import sys

if __name__ == '__main__':

    # Usage: python3 fvecs_to_bin.py file.fvecs file.fbin

    infile = sys.argv[1]
    outfile = sys.argv[2]
    with open(infile, 'rb') as inf:
        data = inf.read()
    nbytes = len(data)
    dim = int.from_bytes(data[0:4], byteorder='little')
    n = nbytes // (dim + 1) // 4
    with open(outfile, 'wb') as outf:
        outf.write(n.to_bytes(4, byteorder='little'))
        outf.write(data[:4])
        for i in range(n):
            outf.write(data[i * (dim + 1) * 4 + 4 : (i + 1) * (dim + 1) * 4])