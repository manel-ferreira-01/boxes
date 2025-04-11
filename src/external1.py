# Python module with the processing to be done in the component
#input (datafile)- bytes with the input mat file
#return (f.getvalue()) - bytes with the output mat file

import numpy as np
import io
import cv2
from scipy.io import loadmat, savemat

# returns numpy arrays with image annotated, xy coordinates of sift and descriptors
def calling_function(datafile):
    #Load the mat file using scipy.io.loadmat
    mat_data=loadmat(io.BytesIO(datafile))
# SPECIFIC CODE STARTS HERE
    im2=mat_data["im"]
    gray= cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints,descriptor = sift.detectAndCompute(gray,None)    
    im2=cv2.drawKeypoints(im2,keypoints,0,(255,0,0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Extract attributes from keypoints
    keypoints_data = []
    for kp in keypoints:
        keypoints_data.append([
            kp.pt[0],  # x coordinate
            kp.pt[1],  # y coordinate
            kp.size,   # size of the keypoint
            kp.angle,  # angle of the keypoint
            kp.response,  # response by which the strongest keypoints have been selected
            kp.octave,  # octave (pyramid layer) from which the keypoint has been extracted
            kp.class_id  # object class (if applicable)
        ])

    # Convert to NumPy array
    keypoints_array = np.array(keypoints_data)
# SPECIFIC CODE ENDS HERE
    f=io.BytesIO()
    savemat(f,{"im":im2,"kp":keypoints_array,"desc":descriptor})
    return f.getvalue()
