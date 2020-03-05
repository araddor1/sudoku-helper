import numpy as np
import cv2
import math

def print_board(board):
        for i in range(len(board)):
            if (i%3 == 0 ):
                print ("-----------------------")
            for j in range(len(board[0])):
                if (j%3 ==0 and j!=0) :
                    print (" | " ,end="")
                print(board[i][j] , end=" ")
            print("\n")
        return
    
def find_empty_spot(board,next):
        for i in range(len(board)):
            for j in range(len(board)):
                if(board[i][j]==0):
                    next[0]=i
                    next[1]=j
                    return True
        return False
    
def legal_in_row(board,row_num,num):
        for i in range(len(board)):
            if(board[row_num][i]==num):
                    return False
        return True

def legal_in_col(board,col_num,num):
        for i in range(len(board)):
            if(board[i][col_num]==num):
                    return False
        return True

def legal_in_box(board,row,col,num):
        sqrt_len=int(math.sqrt(len(board)))
        base_row = row - row%sqrt_len
        base_col = col- col%sqrt_len
        for i in range(sqrt_len):
            for j in range(sqrt_len):
                if(board[base_row+i][base_col+j]==num):
                    return False
        return True           

def legal_pos(board,row,col,num):
        return legal_in_col(board,col,num) and legal_in_row(board,row,num) and legal_in_box(board,row,col,num)

def solve_sudoko(board):
        next = [0,0]

        if (not find_empty_spot(board,next)) :
            return True
        
        for num in range(1,len(board)+1):
            if (legal_pos(board,next[0],next[1],num)):
                board [next[0]][next[1]]=num
                if (solve_sudoko(board)):
                    return True
                board [next[0]][next[1]]=0
        
        return False

    
def max_distnace_between_corners(co):
    max=0
    for p1 in range(len(co)):
        for p2 in range(len(co)):
            if (not(p1 == p2 or p1+p2 == 3)):
                dist=math.sqrt(((co[p1][0]+co[p1][1]) ** 2) + ((co[p2][0]+co[p2][1]) ** 2))
                if (dist>max):
                    max=dist
    return max

def clean_img(img):
        ## threshold image
        img_2= cv2.GaussianBlur(img,(5,5),0)
        img_3= cv2.adaptiveThreshold(img_2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,13,5)

        ## cut out things other the main table
        contours ,hirer = cv2.findContours(img_3,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        max_cont= max(contours, key=cv2.contourArea)
        mask = np.zeros_like(img_3)
        cv2.fillPoly(mask,[max_cont],1)
        img_4=np.multiply(img_3,mask)
        
        ## transform to warped table
        rect = cv2.minAreaRect(max_cont)
        box = np.array(sorted(cv2.boxPoints(rect), key = lambda i: (i[0]+ i[1])))
        if (box[1][0] > box[2][0]):
            tmp=box[1].copy()
            box[1]= box[2]
            box[2]=tmp
        max_dist=max_distnace_between_corners(box)
        dst = np.array([[0, 0],[0, max_dist], [max_dist, 0], [max_dist , max_dist]], dtype='float32')
        M = cv2.getPerspectiveTransform(box,dst)
        img_5 =cv2.warpPerspective(img_4, M, (int(max_dist), int(max_dist)))
        return img_5
        
def get_digit_from_img(img,row,col,box_size):
    cur_img = img[row*box_size :(row+1)*box_size, col*box_size : (col+1)*box_size]
    cv2.imshow('final',cur_img)
    cv2.waitKey(0)

def img_to_board(img):
    board = np.zeros([9 , 9])
    mini_box_size = int (math.sqrt(np.size(img))/ 9 )
    for i in range(9):
        for j in range(9):
            board[i][j]=get_digit_from_img(img,i,j,mini_box_size)
    return board
            
        

    
def img_proccess(img):
        im=clean_img(img.copy())
        cv2.imshow('final',im)
        cv2.waitKey(0)
        img_to_board(im)

        
def main():
        board = [
        [7,8,0,4,0,0,1,2,0],
        [6,0,0,0,7,5,0,0,9],
        [0,0,0,6,0,1,0,7,8],
        [0,0,7,0,4,0,2,6,0],
        [0,0,1,0,5,0,9,3,0],
        [9,0,4,0,6,0,0,0,5],
        [0,7,0,3,0,0,0,1,2],
        [1,2,0,0,0,7,4,0,0],
        [0,4,9,2,0,6,0,0,7]
        ]
        
        img = cv2.imread('images/sudo4.jpeg', 0)
        img_proccess(img)
        



if __name__ == "__main__":
    main()
