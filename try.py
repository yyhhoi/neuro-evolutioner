
import numpy as np
import os
from glob import glob


# weights_dir = 'experiment_results/visualisation/delayed_activation/gen-15_species-585/weights'
# weights_paths = sorted(glob(os.path.join(weights_dir, "*")))


# for idx,w_path in enumerate(weights_paths):
    
#     w = np.load(w_path)
#     print("%d: num of nan = %d" % (idx, np.sum(np.isnan(w))))
#     if idx > 30:
#         break


import numpy as np

import skvideo.io as skv
import skvideo.datasets

filename = skvideo.datasets.bigbuckbunny()

vid_in = skv.FFmpegReader(filename)
data = skv.ffprobe(filename)['video']
rate = data['@r_frame_rate']
T = np.int(data['@nb_frames'])
print(type(rate))
print(rate)