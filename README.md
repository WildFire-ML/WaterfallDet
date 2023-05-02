# WaterfallDet
Waterfall Detection Algorithm from LAS

## Usage
To use, navigate to the directory in an environment with all the required packages installed.

Command
```
python run_waterfall.py -h
usage: run_waterfall.py [-h] [--las-file LAS_FILE] [--resolution RESOLUTION] [--window-size WINDOW_SIZE]
                        [--min_height MIN_HEIGHT] [--smooth-factor SMOOTH_FACTOR] [--th TH] [--thStep THSTEP]
                        [--thmin THMIN]

optional arguments:
  -h, --help            show this help message and exit
  --las-file LAS_FILE   path to the las file
  --resolution RESOLUTION
                        resolution to create the dtm and chm
  --window-size WINDOW_SIZE
                        window size to use for local maxima search
  --min_height MIN_HEIGHT
                        minimum height to call a detection a tree
  --smooth-factor SMOOTH_FACTOR
                        kernel to use for gaussian blur on cmm for smoothing
  --th TH               initial threshold for area growing (from paper)
  --thStep THSTEP       threshold step for area growing (from paper)
  --thmin THMIN         minimum threshold for area growing (from paper)
```

## References
[1] A Segmentation-Based Method to Retrieve Stem Volume Estimates from 3-D Tree Height Models Produced by Laser Scanners !(http://vis-www.cs.umass.edu/AerialImage/Forestry/wiki/lib/exe/fetch.php?media=tree_segmentation:lidarsegmentation.pdf) 
