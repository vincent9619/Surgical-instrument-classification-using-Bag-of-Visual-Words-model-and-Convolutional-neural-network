import cv2


def main():

    img_src = cv2.imread("7_Microvascular_Needle_Holder24.JPG")
    img_result1 = cv2.pyrDown(img_src)
    img_result2 = cv2.pyrDown(img_result1)
    #img_result3 = cv2.pyrDown(img_result2)
    print("img_src=", img_src.shape)
    print("img_result1=", img_result1.shape)
    print("img_result2=", img_result2.shape)
    #print("img_result3=", img_result3.shape)
    cv2.imwrite("img_src"+"down1.jpg", img_result1)
    cv2.imwrite("img_src"+"down2.jpg", img_result2)
    #cv2.imshow("img_result3", img_result3)
    #cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
