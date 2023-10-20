import numpy as np
import cv2

def stack(images):
    
    res = np.zeros(images[0].shape)
    for img in images:
        res += img
    
    return res

def normalize(img):
    mininum = np.min(img)
    maximum = np.max(img)
    # print(mininum)
    # print(maximum)
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

def fix_black_edge(image, baseline):
    R, C ,CH = np.where(image < 1)
    # print(len(R), len(C), len(CH))
    mask = np.zeros(image.shape)
    for r, c, ch in zip(R, C, CH):
        mask[r, c, ch] = 1
    # mask = cv2.resize(mask, (1920, 1080)) 
    # cv2.imshow("mask", mask * 1)
    # cv2.waitKey()
    mask = mask * baseline
    return image + mask