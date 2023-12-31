# tflow-tests

This project is a prototype of an image classifier and learning model. It uses tensorflow to load images from a database located in the local directory `image-library`, train the model. New candidate pictures that are highly confident that they're copy/pastes get shunted to the image library directory into the correct sub-directory. First draft will have the directory local. Next draft will get a compressed directory from a remote source.

Architecture consists of a loader to split the image library into a training and validation dataset. These two datasets are stored in memory.
