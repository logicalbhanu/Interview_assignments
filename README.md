# Interview_assignments
This repository contains the assignments that i have to solve during my Interviews for Data Scientist position.

The Deployment of Quesiton 1.2, is done with flask with local server, 
to execute, enter the Interview_assignments folder and follow the video instructions.

Answers of Bonus questions :

1. Write about any difficult problem that you solved. (According to us difficult - is
something which 90% of people would have only 10% probability in getting a
similarly good solution).

Ans : For me the training the Dense net model(basically a research paper on Dense net need to be implemented), without using dropout and using less than 1 million parameter over cifr-10 dataset to get the accuracy of more than 94%, is the toughest part 
that i have to solve, 

To solve this i implement the research paper to it exact manner, as it is described in the paper, 
And in the place of Dropout I use BatchNorm and Max pooling as regularisation technique, 
Max pooling works because it also decrease the number of pixels of image we consider for the next cnn layer.


2. Formally, a vector space V' is a subspace of a vector space V if
○ V' is a vector space
○ every element of V′ is also an element of V.
Note that ordered pairs of real numbers (a,b) a,b∈R form a vector space V. Which of
the following is a subspace of V?
● The set of pairs (a, a + 1) for all real a
● The set of pairs (a, b) for all real a ≥ b
● The set of pairs (a, 2a) for all real a
● The set of pairs (a, b) for all non-negative real a,b

Ans: option 2 and option 4 would not be a subspace, because inverse would not exist for them, over addition operation.
for example in the case of 'set of pairs (a, b) for all real a ≥ b', the inverse of (3,1) is (-3, -1), but -3 < -1, thus not the element of the set.

and in the case of 'set of pairs (a, b) for all non-negative real a,b', we don't have negative number and therefore we would not have any inverse with addition operation.
