#!/usr/bin/python3
import getopt
import re
import sys

PATTERN_TIMESTAMP = re.compile('^[0-9][0-9]:[0-9][0-9]:[0-9][0-9],[0-9][0-9][0-9] --> [0-9][0-9]:[0-9][0-9]:[0-9][0-9],[0-9][0-9][0-9]')
PATTERN_NUM = re.compile('\\d+')


def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('srt_worker.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
          if opt == '-h':
             print( 'Usage: convert_bilingual_monolingual.py -i <inputfile> -o <outputfile>')
             sys.exit(-2)
          elif opt in ("-i", "--ifile"):
             inputfile = arg
          elif opt in ("-o", "--ofile"):
             outputfile = arg

    if not inputfile:
        print('no input file is specified.\nUsage: convert_bilingual_monolingual.py -i <inputfile> -o <outputfile>')
    elif not outputfile:
        print('no output file is specified.\nUsage: convert_bilingual_monolingual.py -i <inputfile> -o <outputfile>')
    else:
        process(inputfile, outputfile)


def process(input_file, output):
    """
    Convert bilingual caption file to monolingual caption, supported caption file type is srt.
    """
    line_count = 0
    with open(input_file) as file:
        with open(output, 'a') as output:
            for line in file:
                if line_count == 0:
                    line_count += 1
                    output.write(line)
                elif PATTERN_TIMESTAMP.match(line):
                    line_count += 1
                    output.write(line)
                elif line == '\n':
                    line_count = 0
                    output.write(line)
                else:
                    if line_count == 2:
                        output.write(line)
                    line_count += 1
        output.close()
        print('conversion completed!')


if __name__ == "__main__":
    main(sys.argv[1:])
