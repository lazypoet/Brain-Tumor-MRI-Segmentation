# Brain-Tumor-MRI-Segmentation
This project aims to create state-of-the-art fully automatic tumor segmentation algorithm of brain MRI scans using Neural Nets 

# Background
This project started as my final year MTech dissertation in 2016. The initial idea was motivated by [Sérgio Pereira's](https://www.google.co.in/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=0ahUKEwjdh-yXhOfSAhWGpo8KHc9YCo0QFggeMAE&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel7%2F42%2F4359023%2F07426413.pdf&usg=AFQjCNGFaZIFVbDkqmvaOUC0-FTDW5f8hw&sig2=5oNoRwT70-HcznK8TZPn2Q&bvm=bv.150120842,d.c2I) model of CNN.
The work mainly focuses on HGG, but will soon extend to LGG as well. As of now, I've fully replicated the HGG CNN with some minor changes to the procedure given in the paper.
My current aim is to modify the procedure to improve the results. The modification can be in any part of the pipeline, i.e., preprocessing, CNN or postprocessing.

***This project does not yet contain the scripts for preprocessing and saving the 3d brain images as 2d brain slices, but it has bee done. The file paths assume that this has already been done.***
# Dataset
The project uses the [MICCAI BRATS 2015 training dataset](https://www.smir.ch/BRATS/Start2015). The database was split into 66% training and 34% testing data.

# TBD

1. Patch Extraction
2. Image reconstruction
3. Results and images
