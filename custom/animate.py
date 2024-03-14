import imageio as imageio
import sys
import os
import re

base_dir = sys.argv[1]
prefix = sys.argv[2]

reg = re.compile(prefix+'.*')
files = [k for k in os.listdir(base_dir) if reg.match(k)]

with imageio.get_writer(base_dir + '/' + prefix + '.gif', mode='I') as writer:
    for filename in sorted(files):
        image = imageio.imread(base_dir+ '/' + filename)
        writer.append_data(image)


