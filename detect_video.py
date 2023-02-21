from api import Detector
import cv2 
import torch
import argparse
from PIL import Image
import numpy as np
from tracker import Tracker
import random

#
class InferDetection:
    def __init__(self,model_path, model_name,input_size) -> None:
        cuda = torch.cuda.is_available()
        device = 'cpu'
        if cuda:
            device = 'cuda'
        self.detector = Detector(model_name='rapid',
                    weights_path=model_path,
                    use_cuda=cuda) 
        self.input_size = input_size      
        self.tracker = Tracker(150, 30, 5)
    
    def _detectImage(self,img):
        rgb = np.asarray(img)
        detections = self.detector.detect_one(pil_img=img,input_size=self.input_size, conf_thres=0.3,return_img=False)
        detections = detections.tolist()
        centers = []
        for bb in detections:
            if len(bb) == 6:
                x,y,w,h,a,conf = bb
            else:
                x,y,w,h,a = bb[:5]
                conf = -1
            x1, y1 = int(x - w/2), int(y - h/2)
            x,y = int(x),int(y)
            self.draw_xywha(rgb,x,y,w,h,a)
            ct = (x,y)
            centers.append(ct)
            # cv2.circle(rgb,ct,7,(0,0,255),-1)
        centers = np.asarray(centers)
        self.tracker.update(centers)
        for j in range(len(self.tracker.tracks)):
            if(len(self.tracker.tracks[j].trace)>1):
                x = int(self.tracker.tracks[j].trace[-1][0,0])
                y = int(self.tracker.tracks[j].trace[-1][0,1])
                tl = (x-10,y-10)
                br = (x+10,y+10)
                # cv2.rectangle(rgb,tl,br,(0,0,255),1)
                cv2.putText(rgb,str(self.tracker.tracks[j].trackId), (x-10,y-20),0, 1.2, self.tracker.tracks[j].color,4)
                for k in range(len(self.tracker.tracks[j].trace)):
                    x = int(self.tracker.tracks[j].trace[k][0,0])
                    y = int(self.tracker.tracks[j].trace[k][0,1])
                    cv2.circle(rgb,(x,y), 3, self.tracker.tracks[j].color,-1)
                cv2.circle(rgb,(x,y), 6, self.tracker.tracks[j].color,-1)

        return rgb

    def draw_xywha(self,im, x, y, w, h, angle, color=(255,0,0), linewidth=5):
        '''
        im: image numpy array, shape(h,w,3), RGB
        angle: degree
        '''
        c, s = np.cos(angle/180*np.pi), np.sin(angle/180*np.pi)
        R = np.asarray([[c, s], [-s, c]])
        pts = np.asarray([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
        rot_pts = []
        for pt in pts:
            rot_pts.append(([x, y] + pt @ R).astype(int))
        contours = np.array([rot_pts[0], rot_pts[1], rot_pts[2], rot_pts[3]])
        cv2.polylines(im, [contours], isClosed=True, color=color,
                    thickness=linewidth, lineType=cv2.LINE_4)

    def process_vido(self,video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        writer = cv2.VideoWriter('f_out.mp4',cv2.VideoWriter_fourcc(*'H264'),10,(800,600))
        while 1:
            ret,img = cap.read()
            if not ret :
                break
            im_pil = Image.fromarray(img)
            rgb = self._detectImage(im_pil)
            img = cv2.resize(rgb,(800,600))
            cv2.imshow('Video',img)
            writer.write(img)
            key = cv2.waitKey(27)
            if key == ord('q') :
                break
        writer.release()
        cap.release()
    

def get_help():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode','-m',required=True,help='Inference Mode video,images,folder[v,i,f]')
    parser.add_argument('--video','-v',required=False,help='Video for inference')
    parser.add_argument('--image','-img',required=False,help='Image file for infer')
    parser.add_argument('--input_dir','-dir',required=False,help='Images dir for inference')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_help()
    engine = InferDetection('./weights/pL1_MWHB1024_Mar11_4000.ckpt','Rapid',1024)
    if args.mode == 'video' or args.mode == 'v':
        engine.process_vido(args.video)



        

            
