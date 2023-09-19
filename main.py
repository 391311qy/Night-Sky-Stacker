import numpy as np
import cv2
import os

def stack(images):
    
    res = np.zeros(images[0].shape)
    for img in images:
        res += img
    
    return res

def normalize(img):
    mininum = np.min(img)
    maximum = np.max(img)
    print(mininum)
    print(maximum)
    return (img - mininum) * (255 - 0) / (maximum - mininum) + 0


def perspective(pts1, pts2, img, shape):
    matrix, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    result = cv2.warpPerspective(img, matrix, (shape[1], shape[0]))
    return result

# ORB algorithm for corner detection
def keypoints(gray_img):
    orb = cv2.ORB_create(nfeatures=500)
    kp, des = orb.detectAndCompute(gray_img, None)
    return kp, des

def match_keypoints(pts1,pts2):
    # create BFMatcher object
    bf = cv2.BFMatcher(
        cv2.NORM_HAMMING, 
        crossCheck=True
    )

    matches = bf.match(pts1, pts2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    return matches

# draw keypoint positions on the match_img (output of cv2.drawMatches)
def draw_keypoint_positions(matches, match_img, img_gray, kp0, kp):
    for m in matches[:100]:
        src = kp0[m.queryIdx].pt
        dst = kp[m.trainIdx].pt
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(match_img,
            '%s,%s'%(str(int(src[0])), str(int(src[1]))),
        (int(src[0]), int(src[1])), font, 2,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(match_img,
            '%s,%s'%(str(int(src[0])), str(int(src[1]))),
        (int(dst[0])+img_gray.shape[1], int(dst[1])), font, 2,(255,0,255),2,cv2.LINE_AA)

# visualize match images
def visualize_match(ref, kp0, img_gray, kp, matches):
    match_img = cv2.drawMatches(ref, kp0, img_gray, kp, matches[:100], cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    draw_keypoint_positions(matches, match_img, img_gray, kp0, kp)
    match_img = cv2.resize(match_img, (1920, 1080)) 
    cv2.imshow('Matches', match_img)
    cv2.waitKey()

def clahe(img):
    # The declaration of CLAHE
    # clipLimit -> Threshold for contrast limiting
    clahe = cv2.createCLAHE(clipLimit=5)
    final_img = clahe.apply(img) + 30
    return final_img

def main():
    cwd = os.getcwd()
    fnames = [i for i in  os.listdir(cwd) if i.endswith(".JPG")]
    paths = [os.path.join(cwd, i) for i in fnames]

    ref = cv2.imread(paths[0], cv2.IMREAD_GRAYSCALE)
    # ref = clahe(ref)
    kp0, pts0 = keypoints(ref)
    ref_shape = ref.shape
    
    images = [cv2.imread(paths[0], cv2.IMREAD_COLOR)]
    for path in paths[1:]:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # img_gray = clahe(img_gray)

        kp, pts = keypoints(img_gray)
        matches = match_keypoints(pts0, pts)

        # addtional filtering:
        # add only vertical shift < a threshold
        good_matches = []
        for m in matches:
            src = kp0[m.queryIdx].pt
            dst = kp[m.trainIdx].pt
            if m.distance < 100 and abs(src[1] - dst[1]) < 100:
                good_matches.append(m)
        matches = good_matches

        src_pts = np.float32([kp0[m.queryIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)
        
        img_transform = np.zeros(img.shape)
        img_transform[:,:,0] = perspective(dst_pts, src_pts, img[:,:,0], ref_shape)
        img_transform[:,:,1] = perspective(dst_pts, src_pts, img[:,:,1], ref_shape)
        img_transform[:,:,2] = perspective(dst_pts, src_pts, img[:,:,2], ref_shape)
        images.append(img_transform)
        # images.append(img)
    res = stack(images)
    
    # normalize image
    # res = cv2.normalize(res, 0, 1500*655.36, cv2.NORM_MINMAX)
    res /= len(images)
    res = (res).astype(np.uint8)
    
    # cropping
    c0, c1 = res.shape[0] // 2, res.shape[1] // 2
    ratio = 9/10
    r_pixels = int(ratio*0.5*res.shape[0])
    c_pixels = int(ratio*0.5*res.shape[1])
    res = res[
        c0-r_pixels:c0+r_pixels,
        c1-c_pixels:c1+c_pixels,
        :
    ]

    brightness = 90
    contrast = 150
    res = np.int16(res)
    res = res * (contrast/127+1) - contrast + brightness
    res = np.clip(res, 0, 255)
    res = np.uint8(res)

    

    cwd = os.getcwd()
    cv2.imwrite(os.path.join(cwd, "output/output_c%s_b%s.jpg"%(str(contrast), str(brightness))), res)
    res = cv2.resize(res, (1920, 1080))  
    cv2.imshow("result", res)
    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)
    
    # closing all open windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



