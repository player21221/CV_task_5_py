# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import cv2

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def cornerHarris_demo(src_gray,thresh=100,show_result=False,fi=0):
    # thresh = val
    # Detector parameters
    blockSize = 2
    apertureSize = 3
    k = 0.04
    # Detecting corners
    dst = cv2.cornerHarris(src_gray, blockSize, apertureSize, k)
    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    dst_norm_scaled = cv2.convertScaleAbs(dst_norm)

    points  = []

    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i,j]) > thresh:
                cv2.circle(dst_norm_scaled, (j,i), 5, (0), 2)
                points += [[[i,j]]]

    # Showing the result
    if show_result:
        cv2.namedWindow("corners_window_"+str(fi))
        cv2.imshow("corners_window_"+str(fi), dst_norm_scaled)
    return np.array(points).astype(np.float32)


# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )


# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm')
    # filenames = [
    #     "photo_2022-02-23_03-58-41.jpg",
    #     "photo_2022-02-23_03-58-46.jpg"
    # ]
    filenames = [
        "photo_2022-02-24_04-30-53.jpg",
        "photo_2022-02-24_04-30-56.jpg"
    ]
    resultname = "result"

    imm = [0 for _ in range(len(filenames))]
    pts = [0 for _ in range(len(filenames))]
    ofl = [0 for _ in range(len(filenames)-1)]
    stm = [0 for _ in range(len(filenames)-1)]
    erm = [0 for _ in range(len(filenames)-1)]

    for f in range(len(filenames)):
        imm[f] = cv2.imread(filenames[f])
        imm[f] = cv2.cvtColor(imm[f],cv2.COLOR_BGR2HSV)
        # pts[f] = cornerHarris_demo(imm[f][:,:,2],100,False,fi=f)
        pts[f] = cv2.goodFeaturesToTrack(imm[f][:,:,2], mask = None, **feature_params)
    ofl_good=[0]*(len(filenames)-1)
    # cv2.waitKey()
    for f in range(len(filenames)-1):
        ofl[f], stm[f], erm[f] = cv2.calcOpticalFlowPyrLK(imm[f][:,:,2],imm[f+1][:,:,2],pts[f],None, **lk_params)
        ofl_good[f]=ofl[f]#[stm[f]==1]
        max_err=np.max(erm[f])
        hom, st = cv2.findHomography(pts[f][:,0,:],ofl_good[f],cv2.RANSAC,max_err,None,100)
        result = cv2.warpPerspective(imm[f],hom,imm[f].shape[::-1][1:3])
        cv2.namedWindow("result_"+str(f))
        cv2.imshow("result_"+str(f), cv2.cvtColor(result,cv2.COLOR_HSV2BGR))
        cv2.imwrite("result_" + str(f) + ".jpeg",cv2.cvtColor(result,cv2.COLOR_HSV2BGR))

    for i, f in enumerate(filenames):
        cv2.namedWindow("input_" + str(i))
        cv2.imshow("input_" + str(i), cv2.cvtColor(imm[i],cv2.COLOR_HSV2BGR))

    # o = np.ones_like(imm[0][:,:,2])*255
    # for i in range(ofl_good[0].shape[0]):
    #     cv2.circle(o, (int(ofl_good[0][i][0]),int(ofl_good[0][i][1])), 5, (0), 2)
    # cv2.namedWindow("o")
    # cv2.imshow("o", o)


    cv2.waitKey()








# See PyCharm help at https://www.jetbrains.com/help/pycharm/
