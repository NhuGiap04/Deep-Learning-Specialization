# each output line should be:
# INPUT<tab>RESPONSE
with open('../large_files/twitter_tab_format.txt', 'w') as f:
    prev_line = None
    # data source: https://githu  b.com/Phylliida/Dialogue-Datasets
    for line in open('../Datasets/TwitterLowerAsciiCorpus.txt'):
        line = line.rstrip()

        if prev_line and line:
            f.write("%s\t%s\n" % (prev_line, line))

        # note:
        # between conversations there are empty lines
        # which evaluate to false

        prev_line = line
