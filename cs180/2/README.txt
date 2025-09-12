Explanation: This project takes the digitized Prokudin-Gorskii glass plate images and, using image processing techniques, 
            automatically produces a color image with as few visual artifacts as possible. 
            In order to do this, I extracted the three color channel images, placed them on top of each other, 
            and aligned them so they formed a single RGB color image.

In order to run the code, running main.py directly will run the image pyramid speedup scheme using normalized cross-correlation 
as the metric. In order to switch back to the L2 norm, change the function being called in the align function from ncc to l2norm, 
make the score +float('inf'), and switch the sign of the if statement from > to <. To remove the image pyramid speedup, replace where 
its being called from pyramid_speedup() to align().
