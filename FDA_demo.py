import numpy as np
from PIL import Image
from utils import FDA_source_to_target_np
import os 

src = "sample_data/omniverse/rgb_0000.png"  #source omniverse image
trg = "sample_data/original/1_7000_0.png"   #target original image

im_src = Image.open(src).convert('RGB')
im_trg = Image.open(trg).convert('RGB')

# if needed to resize image 
im_src = im_src.resize( (2048,1024), Image.BICUBIC )
im_trg = im_trg.resize( (2048,1024), Image.BICUBIC )

im_src = np.asarray(im_src, np.float32) / 255.0
im_trg = np.asarray(im_trg, np.float32) / 255.0

im_src = im_src.transpose((2, 0, 1))
im_trg = im_trg.transpose((2, 0, 1))

# Add this after loading images
# print("Source range:", im_src.min(), im_src.max())
# print("Target range:", im_trg.min(), im_trg.max())

# L_list = [0.001,0.002,0.003,0.004,0.005,0.006,0.008,0.01,0.03,0.05,0.08,0.1,0.2,0.5,0.7,0.8]
L_list = [0.001, 0.0005, 0.0003] 

for l in L_list:
    src_in_trg = FDA_source_to_target_np( im_src, im_trg, L=l)
    print(l)
    src_in_trg = src_in_trg.transpose((1,2,0))
    src_in_trg = np.clip(src_in_trg * 255.0, 0, 255).astype(np.uint8)

    output_name = "FDA_output" 

    if not os.path.exists(output_name):
        os.makedirs(output_name)
    file_path = os.path.join(output_name, f'L_{l}.png')
    Image.fromarray(src_in_trg).save(file_path)
    print(f'L_{l}.png generated')