## BaseLine

## BottomRowUnifier

## CaMeasurer

## DropDetection
In this directory you can find 3 separate method to detect a drop inside a frame.
1. Difference method and its derivatives:
    1. In the 4S-SROF Sajjad had implemented an absolute difference and then by parameters designed by hand found the drop.
    2. With Difference method by considering the two consecutive drops and calculating the difference boundaries of the drop is reveled.

2. Using histogram to find the variation in pixels color. The fastest method and easiest. Only calculate the width of of drop and doesn't care about the height information. 

3. YOLO, most expensive and interesting one. But I don't use it anymore. For whole dataset it takes around 10 hours yet summations/histogram takes 1 hour.