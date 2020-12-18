# Computer_Vision3-Image-Style-Transfer
This project is the core stone of a mobile application to enhance the experience when visiting a museum or an art gallery . The visitor can take a shot for a painting and then the program would extract the painting without the frame, apply the desired image style transfer and then place the edited painting inside its frame in the original image.

It can be divided into several steps:

1- Detect the painting.

2- Extract it and project it using Projective transform

3- Apply the image style transfer using VGG19 network

4- Place the image back in its original location using Inverse Projective trnasform.

5- Detection evaluation was computed using IuO metric. Overall IuO accurracy on test set was 88%
