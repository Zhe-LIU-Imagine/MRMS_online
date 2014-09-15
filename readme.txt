                   MRMS Library
                     Zhe Liu

ABOUT
  The MRMS open source library implements the match refinement and match selection method introduced in the paper with minor modifications: 

Z,Liu; P.Monasse and R,Marlet. Match Selection and Refinement for Highly Accurate Two-View Structure from Motion in ECCV 2014

  It is distributed under the BSD license (see the COPYING file).

PERFORMANCE WARMING:
  1. It may be time consuming to run the whole test for a scene(eg. 7 pairs of image for 4 methods and 16 iterations). The result of the input data is already in the output folder.

  2. If you use openCV library, please use version later than 2.4, otherwise descriptors' orientation may be reversed! (Bug in openCV 2.3) An easy way to test is to process an image with its rotated projection.


INSTALLING
  This implementation is on C++ and depends on openCV and KVLD library, whose installation guild is available online. You will need Cmake 2.6 or later to compile the program.
  
FOLDERS:
  Kvld: containing all KVLD algorithm, some structures depend on
   OrsaHomography library, so please include both of them to make KVLD running.
  OrsaHomography: containing ORSA algorithm implemented by Pierre Moulon. It also offers basic structures for KVLD algorithm.
  Input: some illustrating pairs of images with ground truth.
  Output: results of demos are sent here, including 
    (? means the image index)
    * ?_(?+1)_log: details of test
    * ?_(?+1)_trueP: ground truth information
    * average_result: final result for each method
    * ?_(?+1)_rotation info: position information for this pair.
    
APPLICATIONS

  The code contains the main applications (MRMS_comparison.cpp). It shows the result of comparison for RANSAC, Match Selection, Match Refinement and MR+MS.

For more information, please contact zhe.liu@enpc.fr



