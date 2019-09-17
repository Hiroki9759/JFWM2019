import cv2
import math

if __name__ == "__main__":
    # input image
    img = cv2.imread("/Users/iwanahiroki/JFWM2019/coin_input-1024x768.jpg")

    # convert gray scale image
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # gaussian
    gray_img = cv2.GaussianBlur(gray_img, (7,7), 0)
    ret, bw_img = cv2.threshold(gray_img, 190, 255, cv2.THRESH_BINARY_INV)
    #ret, bw_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)

    # invert black white (when use cv2.THRESH_OTSU)
    #bw_img = cv2.bitwise_not(bw_img)

    cv2.imwrite("black_white.jpg", bw_img)

    imgEdge, contours, hierarchy = cv2.findContours(bw_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ROUNDNESS_THRESH = 0.5
    ASPECT_THRESH = 0.90
    COIN_AREA = 100000.0
    COIN_AREA_RATIO_THRESH = 0.5
    coin_list = []
    for contour in contours:
        # get circumscribed quadrangle
        x, y, width, height = cv2.boundingRect(contour)

        # check aspect ratio
        aspect_ratio = float(width) / float(height)
        if (aspect_ratio < ASPECT_THRESH):
            continue

        # check area
        area = cv2.contourArea(contour)
        area_ratio = abs(float(1 - (area / COIN_AREA)))
        if (area != 0 and area_ratio > COIN_AREA_RATIO_THRESH):
            continue

        # detect long axis
        longAx = width
        if (width < height):
            longAx = height
        # calculate roundness value
        roundness = (4*area)/(math.pi*(longAx**2)) # it seems like a circle closer to 1.0

        if (roundness > ROUNDNESS_THRESH):
            coin_list.append(roundness)
            topleft = x
            cv2.rectangle(img, (x, y), (x+width, y+height), (0, 0, 200), 2)

    print("number of coins detected : ", len(coin_list))
    print("coin average roundness : ", sum(coin_list)/len(coin_list))
    cv2.imwrite("coin_result.jpg", img)