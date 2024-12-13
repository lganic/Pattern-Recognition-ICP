This directory contains utilities used to parse, filter, and segment the MPII and COCO dataset. 

- coco-read : Read the data, and metadata from coco and output to file - json pairs.
- mpii-read : Read the data, and metadata from mpii and output to file - json pairs.
- filter_body : Apply filtering to isolate and crop images to effective sizes
- filter_nopoint : Remove files that have no person contained in them
- generate_heatmap : Loop over all files in a directory, and create a new dataset with the image and a produced heatmap 
- heatmap_split : Split a heatmap directory into test / train /val sets. 
- test_heatmap : Display the generated heatmaps
