# load the existing model
# perform standard transfer learning
# use binary crossEntropy: 0 is handwritten, 1 is computer generated
# it makes sense to have the output layer be Dense(1, activation='sigmoid'), and the
#    previous layer correspond to all the different ways computer generated lines can exist
# I expect there to be at least 1 hidden layer, with an unknown number of units.  Use keras
#    tuner to discover the correct number of nodes. 