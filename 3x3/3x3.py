import numpy as np
import cv2
import Queue
from matplotlib import pyplot as plt

#이미지 라벨링 함수
def labeling(height, width, image):
    #라벨링 할 행열 선언
    lab_img= [[0 for col in range(height)] for row in range(width)]

    #원본 이미지 행열을 복사하여 라벨링 할 행열에 넣어줌
    for i in range(0, height):
        for j in range(0, width):
            if (image[i][j] == 255):
                lab_img[i][j] = -1 #라벨링이 되지 않은 값이므로 1이 아닌 -1로 저장함
            if (image[i][j] == 0):
                lab_img[i][j] = 0
            if (j == 0 or j == width-1 or i == 0 or i == height-1): #영상 바깥으로 나가는 것 방지하기 위해 경계 값은 0으로 설정
                lab_img[i][j] = 0

    label=1 #라벨링 번호는 1부터 시작

    #라벨링 할 행열을 검사함. 영상 바깥으로 나가는 것을 방지하기 위해 1부터 높이-2, 폭-2 까지 검사함
    for i in range (1, height-1):
        for j in range (1, width-1):
            if (lab_img[i][j] == -1): #번호가 붙여져 있지 않으면
                efficient_floodfill4(lab_img,i,j,label) #4연결성 검사
                label += 1 #다음 라벨링 번호


#4연결성 범람 채움 함수. 스택 오버플로우를 피하는 법
def efficient_floodfill4(lab_img,i,j,label):

    #비어있는 큐 생성
    q = Queue.Queue()

    #큐에 검사 할 원소 넣어줌
    q.put((i,j))

    while (q.empty() == False) :

        #선언
        (y,x) = (0,0)

        #큐에서 원소 하나 꺼냄
        (y,x) = q.get()

        if (lab_img[y][x] == -1) : #번호가 붙여져 있지 않으면
            #선언
            left = right = x

            #아직 미처리 상태의 열 찾음. 양 옆 검사
            while (lab_img[y][left-1] == -1):
                left -= 1
            while (lab_img[y][right+1] == -1):
                right += 1

            #left부터 right까지 검사해줌
            for c in range(left, right+1):
                lab_img[y][c] = label #라벨링 해줌
                #위아래 좌표에 번호가 없으면 위아래 좌표를 큐에 추가해서 그 열도 검사할 수 있도록 함
                if (lab_img[y-1][c] == -1 and (c==left or lab_img[y-1][c-1]!=-1)):
                    q.put((y-1,c))
                if (lab_img[y+1][c] == -1 and (c==left or lab_img[y+1][c-1]!=-1)):
                    q.put((y+1,c))


#팽창 함수
def dilation(height, width, image):
    #팽창 할 이미지 복사
    dil_img = image.copy()

    for i in range(0, height):
        for j in range(0, width):
            if (image[i][j] == 255):
                image[i][j] = 1 #검사를 위해 1로 바꿔줌
                dil_img[i][j] = 1
            if (j == 0 or j == width - 1 or i == 0 or i == height - 1): #영상 바깥으로 나가는 것 방지하기 위해 경계 값은 0으로 설정
                dil_img[i][j] = 0

    #3X3 크기의 윈도우 만들어줌
    window = [[1 for col in range (3)] for row in range (3)]

    #팽창 할 원본 행열을 검사함. 영상 바깥으로 나가는 것을 방지하기 위해 1부터 높이-2, 폭-2 까지 검사함
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            #현재 검사 요소 값이 1이면 필터에 해당하는 범위를 모두 1로 바꾸어줌. 4연결성이기 때문에 대각선은 변경해주지 않음
            if (image[i][j] == window[1][1]):
                #영상 출력을 위해 1이 아닌 255로 바꿔줌
                dil_img[i-1][j] = 255
                dil_img[i][j-1] = 255
                dil_img[i][j] = 255
                dil_img[i][j+1] = 255
                dil_img[i+1][j] = 255


    #이미지 출력
    cv2.imshow("dilation image", dil_img)

    #esc키를 눌러야만 이미지가 사라지고 프로그램 종료
    while 1:
        key = cv2.waitKey(0)
        if key == 27 : #esc key
            cv2.destroyAllWindows()
            break




#침식 함수
def erosion(height, width, image):
    #침식 할 이미지 복사
    ero_img = image.copy()

    for i in range(0, height):
        for j in range(0, width):
            if (image[i][j] == 255):
                image[i][j] = 1 #검사를 위해 1로 바꿔줌
                ero_img[i][j] = 1
            if (j == 0 or j == width - 1 or i == 0 or i == height - 1): #영상 바깥으로 나가는 것 방지하기 위해 경계 값은 0으로 설정
                ero_img[i][j] = 0

    #3X3 크기의 윈도우 만들어줌
    window = [[1 for col in range (3)] for row in range (3)]

    #침식 할 원본 행열을 검사함. 영상 바깥으로 나가는 것을 방지하기 위해 1부터 높이-2, 폭-2 까지 검사함
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            #4연결 중 모두 1일 때만 현재 위치 값을 1로 바꾸어줌. 4연결성이기 때문에 대각선은 검사해주지 않음
            if (image[i - 1][j] == window[0][1] and image[i][j - 1] == window[1][0] and image[i][j] == window[1][1]
                and image[i][j + 1] == window[1][2] and image[i + 1][j] == window[2][1]):
                #영상 출력을 위해 1이 아닌 255로 바꿔줌
                ero_img[i][j] = 255

            else:
                ero_img[i][j] = 0

    #이미지 출력
    cv2.imshow("erosion image", ero_img)

    #esc키를 눌러야만 이미지가 사라지고 프로그램 종료
    while 1:
        key = cv2.waitKey(0)
        if key == 27:  # esc key
            cv2.destroyAllWindows()
            break


#image 불러옴
img = cv2.imread('Image1.pgm')

#회색 영상으로 변환
image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#이미지 정보(높이, 폭) 받아옴
height, width = image.shape

labeling(height, width, image)
dilation(height, width, image)
erosion(height, width, image)


