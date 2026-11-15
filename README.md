# Hidden in Plain Sight 
## Submission from team: Raspy Giant Hamster 
An HP-protein lattice inspired approach to text encoding 

## installation
clone the git repository on one or more devices:
`git clone https://github.com/JDBK16/Durhack2026.git`
`py -m pip install cv2`
`py -m pip install matplotlib`
`py -m pip install openCV`
`py -m pip install numpy`

## operation 

To run the we recommend running `pipeline.py -r True` or `pipeline.py -random True` 

The pipeline runs through the entire process of encoding three randomly selected words from the 'durhack_word_list.csv' file (in the data directory), creating a visual plot of the encoded image, adding said image to the proteins directory, parsing the image with the computer vision pipeline and printing the decoded words in order.

alternatively, the individual files for each step of the process may be run seperately in the following order (this is the best method for using the video capture mode for the graph scannner):
`py main.py`
then on a seperate device (for video or still images respectively):
`py graph_scanner.py --video` or `py graph_scanner.py --snapshot`
then run
`py testing.py`
if there is no output, the testing suite has run successfully 
