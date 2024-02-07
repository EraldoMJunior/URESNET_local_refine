#!/bin/bash


python inference.py examples/cat-1460887_1280_img.jpg  examples/cat-1460887_1280_mask.png  examples/cat-1460887_1280_out.png --background=black
python  inference.py examples/00081-958218434_img.jpg examples/00081-958218434_mask_extra.png examples/00081-958218434_out.jpg
python  inference.py examples/simg_089.jpg examples/smask_089.png examples/output_089.jpg

cd examples
montage cat-1460887_1280_img.jpg  cat-1460887_1280_mask.png cat-1460887_1280_out.png -geometry +0+0 cat_compose.jpg
montage simg_089.jpg smask_089.png output_089.jpg -tile 3x1 -geometry +0+0 s_compose.jpg
montage 00081-958218434_img.jpg 00081-958218434_mask_extra.png 00081-958218434_out.jpg -tile 3x1 -geometry +0+0 00081-958218434_compose.jpg

cd ..