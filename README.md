To train the neural network, run OCR.py. If the EMNIST balanced dataset is not yet downloaded, 
set download = True in line 29 of the file. The network will be trained, outputting the training accuracy per epoch.
At the end of the training the model is saved in your working directory.
The testing section is in testing.py, again set download = True in line 22 if the data is not yet downloaded. 
First the model will be tested on EMNIST balanced test data, of which the result will be outputted.
We included our own images img1.jpg - img27.jpg for testing together with images.txt for the labels.
If you want to add your own images, you can add them as "imgX.jpg" and add a line to images.txt: 
"X, " and then the image label (without any spaces!) 
make sure the label excists in EMNIST balanced: eg "z" is replaced by "Z"
Images should be jpg format, upright and made so the character is darker than the background

When training the own images all the wrongly predicted images get printed: real label / predicted label.
It is also possible to uncomment code on 112-117 to display every image along with real label/predicted label