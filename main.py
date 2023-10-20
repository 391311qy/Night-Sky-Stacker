import numpy as np
import cv2
import os
from utils import *

class Sticher(object):

    def __init__(self, sequence_dir=None):
        self.brightness = 90
        self.contrast = 150
        self.match_dist_max=100
        self.match_dist_vert_max=100
        self.cwd = os.getcwd() if not sequence_dir else sequence_dir
        fnames = [i for i in  os.listdir(self.cwd) if i.endswith(".JPG")]
        self.paths = [os.path.join(self.cwd, i) for i in fnames]

    def filter_matches(self, matches, kp0, kp):
        """
            Additional filtering of the keypoints, 
            here we select those matches that has a vertical distance smaller
            than a threshold as the good match. 
        """
        good_matches = []
        for m in matches:
            src = kp0[m.queryIdx].pt
            dst = kp[m.trainIdx].pt
            if m.distance < self.match_dist_max and abs(src[1] - dst[1]) < self.match_dist_vert_max:
                good_matches.append(m)
        return good_matches
    
    def perspective_transform(self, img, ref_shape, matches, kp0, kp):
        src_pts = np.float32([kp0[m.queryIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)
        
        img_transform = np.zeros(img.shape)
        img_transform[:,:,0] = perspective(dst_pts, src_pts, img[:,:,0], ref_shape)
        img_transform[:,:,1] = perspective(dst_pts, src_pts, img[:,:,1], ref_shape)
        img_transform[:,:,2] = perspective(dst_pts, src_pts, img[:,:,2], ref_shape)
        return img_transform
    
    def normalize(self, res, nimages):
        res /= nimages
        res = (res).astype(np.uint8)
        return res

    def adjust_brightness_and_contrast(self, res):
        contrast = self.contrast
        brightness = self.brightness
        res = np.int16(res)
        res = res * (contrast/127+1) - contrast + brightness
        res = np.clip(res, 0, 255)
        res = np.uint8(res)
        return res

    def write_out(self, cwd, res):
        path = os.path.join(cwd, "output/output_fixed_c%s_b%s.jpg"%(str(self.contrast), str(self.brightness)))
        print("write to %s"%path)
        cv2.imwrite(path, res)
        # res = cv2.resize(res, (1920, 1080))  
        # cv2.imshow("result", res)
        # # waits for user to press any key
        # # (this is necessary to avoid Python kernel form crashing)
        # cv2.waitKey(0)
        
        # # closing all open windows
        # cv2.destroyAllWindows()

    def process(self):
        ref = cv2.imread(self.paths[0], cv2.IMREAD_GRAYSCALE)
        kp0, pts0 = keypoints(ref)
        ref_shape = ref.shape

        images = [cv2.imread(self.paths[0], cv2.IMREAD_COLOR)]
        for path in self.paths[1:]:

            print("working on %s ..." %path)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # img_gray = clahe(img_gray)
            kp, pts = keypoints(img_gray)
            matches = match_keypoints(pts0, pts)

            # addtional filtering:
            # add only vertical shift < a threshold
            matches = self.filter_matches(matches, kp0, kp)
            
            # visualize_match(ref, kp0, img_gray, kp, matches)

            # perform perspective transform on 3 channels
            img_transform = self.perspective_transform(img, ref_shape, matches, kp0, kp)
            
            # fix black edges due to transform
            img_transform = fix_black_edge(img_transform, images[0])
            images.append(img_transform)
            # images.append(img)
        res = stack(images)
        
        # normalize image
        # res = cv2.normalize(res, 0, 1500*655.36, cv2.NORM_MINMAX)
        res = self.normalize(res, len(images))
        res = self.adjust_brightness_and_contrast(res)
        return res

def main():

    sticher = Sticher(os.getcwd())
    res = sticher.process()
    sticher.write_out(os.getcwd(), res)
    

if __name__ == "__main__":
    main()



