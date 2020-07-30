To use the code:
mapping =getmapping02(n,'u2');
%       'u2'   for uniform LBP
%       'ri'   for rotation-invariant LBP
%       'riu2' for uniform rotation-invariant LBP.
OurLBP(IMAGE,th,n,'el',p1,p2,beta,mapping,'nh');

•	th is the threshold, if you want to use a binary coding set: th=[];
•	n is the number of neighbours
•	p1 and p2 are the parameters of the loci of points (see Table 1), 
    e.g. p1 is the radius if you use “circle”; p1 and p2 are the semimajor 
    and semiminor axis lengths if you use an ellipse
•	the fourth input permits to select the loci: 'el', ellipse; 'ip', 
    hyperbole;'par', parabola; 'sp', spiral; 'ci', circle.
•	Beta is the rotation angle

These approaches are suited for feeding a robust classifier, we use SVM. 
Also in the literature almost in all the problems LTP outperforms LBP, when
 they are used for training SVM. If you use a distance NOT always this 
happen, due to the curse of dimensionality problem. 

Regards
L
