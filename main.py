import os

import cv2
import face_recognition
import numpy as np
import open3d as o3d
from PIL import Image

from accelerate import Accelerator
from transformers import pipeline


device = Accelerator().device
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN")

pipe = pipeline(
    "depth-estimation",
    device = -1,
    model="depth-anything/Depth-Anything-V2-Small-hf",
)

previous_depths = {}
test= True
def find_depth(found_object,face_id):
    if test is not False:
        #demo code for showing off
        if found_object is None or found_object.size == 0:
            return None
        # convert grey and to np array 
        found_object= cv2.cvtColor(found_object, cv2.COLOR_BGR2GRAY)
        found_object= found_object.astype(np.uint8)
        #return faux depthmap of image
        depth_heatmap = cv2.applyColorMap(found_object, cv2.COLORMAP_JET)
        return depth_heatmap
    

    global previous_depths
    if found_object is None or found_object.size == 0:
        return None
    ##make predictions upon the image
    predictions = pipe(Image.fromarray(cv2.cvtColor(found_object, cv2.COLOR_BGR2RGB)))
    depth_map = np.array(predictions["depth"], dtype=np.float32)
    #get the previous ID
    prev = previous_depths.get(face_id)
    if prev is None or prev.shape != depth_map.shape:
        previous_depths[face_id] = depth_map
    else:
        alpha_smooth = 0.4
        depth_map = alpha_smooth * prev + (1 - alpha_smooth) * depth_map
        previous_depths[face_id] = depth_map
    depth_map = depth_map.astype(np.uint8)
    depth_heat = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    return depth_map, depth_heat


count = 0
fc = 1
cap = cv2.VideoCapture({Enter Video File Name here})
source_fps = cap.get(cv2.CAP_PROP_FPS)/fc
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()
#get resolution    
src_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
src_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
target_max = 120
##if width is bigger go on width else go on height for verticle
if src_width >= src_height:
    scale = target_max / src_width
else:
    scale = target_max / src_height
width = int(src_width * scale)
height = int(src_height * scale)

video = cv2.VideoWriter("test.mp4", cv2.VideoWriter_fourcc(*'mp4v'), source_fps, (width, height))

alpha = 0.6
while success:
    count+=1
    if count % fc != 0:
        success = cap.grab() # Fetches frame from buffer but DOES NOT decode it
        if not success:
            break
        continue # Instantly skip to the next loop iteration
    success, image = cap.read() # Read frame
    if success:
        if not success:
            break
        image = cv2.resize(image,(width, height), interpolation=cv2.INTER_NEAREST)
        face_locations = face_recognition.face_locations(image)    
        for i,faces in enumerate(face_locations):
            # get face location
            top, right, bottom, left = faces
            face = image[top:bottom, left:right] 
            if not test:
                face_depth, face_heat = find_depth(face, i)#apply depth of the found object
            else: face_heat= find_depth(face, i)
            
            if face_heat is not None:
                ##apply depth filter upon the face image
                blended = cv2.addWeighted(face_heat, alpha, image[top:bottom, left:right], 1 - alpha, 0) 
                image[top:bottom, left:right] = blended
        ##write video and show image
        video.write(image)
        cv2.imshow('Depth Feed',image)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):  # Press 'q' to quit
            break
cv2.destroyAllWindows()
video.release()
