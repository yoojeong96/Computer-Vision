import cv2;
import numpy as np
from matplotlib import pyplot as plt


#image 정보로 히스토그램 그림
def histogram(height, width):

    #크기 256 배열 0으로 초기화
    histo = [0.0] * 256;

    for i in range(0, height):
        for j in range(0, width):
            histo[ image[i][j] ] += 1;

    #히스토그램 평균값 구함
    avg = 0;

    for i in range(0, 256):
        avg += i * histo[i];

    avg /= width * height;

    #정규화
    for i in range(0, 256):
        histo[i] /= width * height;

    return histo, avg;


#왼쪽과 오른쪽 사이의 거리의 최대값을 구함
def otsu():
    histo, avg = histogram(height, width);

    threshold = 0;
    v0 = 0;

    #크기 256 배열 0으로 초기화
    w0 = [0.0] * 256;
    w0[0] = histo[0];

    #크기 256 배열 0으로 초기화
    u0 = [0.0] * 256;

    for t in range(1, 256):
        #히스토그램 누적된 값
        w0[t] = w0[t - 1] + histo[t];

        #분모에 0이 들어갈 수 없음
        if (w0[t] == 0.0):
            continue;

        #왼쪽 평균
        u0[t] = (w0[t - 1] * u0[t - 1] + t * histo[t]) / w0[t];

        #분모에 0이 들어갈 수 없음
        if (1 - w0[t] == 0.0):
            continue;

        #오른쪽 평균
        u1 = (avg - w0[t] * u0[t]) / (1 - w0[t]);

        v1 = w0[t] * (1 - w0[t]) * pow((u0[t] - u1), 2);

        #왼쪽과 오른쪽이 비슷해질때 threshold 값 얻음
        if (v1 > v0):
            v0 = v1
            threshold = t;

    return threshold;


#otsu algorithm에서 얻은 threshold 값을 이용한 image binarization
def binarization(threshold, image):

    threshold = otsu();

    #pixel값이 threshold 보다 크면 255, 작으면 0으로 변환
    for i in range(0, height):
        for j in range(0, width):
            if (image[i][j] >= threshold):
                image[i][j] = 255;
            else :
                image[i][j] = 0;

    #threshould 값과 변환된 이미지 출력
    print(threshold);
    cv2.imshow("Otsu image", image);
    cv2.waitKey(0);
    cv2.destroyAllWindows();



#####################################

img = cv2.imread('cameraman.png', cv2.IMREAD_COLOR);

#image의 높이, 폭 저장. color가 있어 channel까지 받아줌
height, width, channel = img.shape;

#image를 회색으로 변환
image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);

histogram(height, width);
threshold = otsu();
binarization(threshold, image);