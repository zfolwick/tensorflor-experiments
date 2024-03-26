# hello-transfer

A hello world for neural network transfer learning on real/computer-generated
images. This builds on hello-mnist's digit recognition neural net and indeed
uses the model generated from it.

This model- instead of recognizing a digit- will instead be directed towards
learning the difference between handwritten digits and computer generated
digits. Ideally, this will be extended to computer generated letters as well,
though that will take some deeper understanding of what is being seen at each
layer of the neural network.

Once this discriminator can effectively determine whether a digit it's trained
on is handwritten, it will likely need to be re-trained with more computer
generated data- this time, any letter given should give a valid prediction as to
whether it's handwritten or not.

The roadmap is:
[ DONE ] successfully ID digits 0 - 9 (hello world example)
[ DONE ] identify handwritten vs. computer generated digits 0 - 9 (hello fake world)
[ ] Identify hand drawn vs computer generated line drawings
[ ] Identify a photograph vs digital art
[ ] Identify digital insertions within a photo taken with a camera - (image segmentation)
[ ] Identify deepfakes
