import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import math
import operator
import sudoko_solver

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)        
        

def dist(p1,p2):
    return math.sqrt(((p1[0]+p1[1]) ** 2) + ((p1[0]+p2[1]) ** 2))

def to_point_convert(max_input,polygon):
    return polygon[max_input[0]][0]
    
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
        bottom_r =to_point_convert(max(enumerate([pt[0][0] + pt[0][1] for pt in max_cont]), key=operator.itemgetter(1)),max_cont)
        top_l = to_point_convert(min(enumerate([pt[0][0] +pt[0][1] for pt in max_cont]), key=operator.itemgetter(1)),max_cont)
        bottom_l =to_point_convert(max(enumerate([pt[0][0] - pt[0][1] for pt in max_cont]), key=operator.itemgetter(1)),max_cont)
        top_r =to_point_convert(min(enumerate([pt[0][0] - pt[0][1] for pt in max_cont]), key=operator.itemgetter(1)),max_cont)
        
        box = np.array([top_l,top_r,bottom_l,bottom_r], dtype='float32')
        max_dist=int(max (dist(top_l,top_r),dist(top_r,bottom_r),dist(top_l,bottom_l),dist(bottom_l,bottom_r)))
        dst = np.array([[0, 0],[0, max_dist-1], [max_dist-1, 0], [max_dist-1 , max_dist-1]], dtype='float32')
        M = cv2.getPerspectiveTransform(box,dst)
        img_5 =cv2.warpPerspective(img_4, M, (int(max_dist), int(max_dist)))
        return img_5

def return_digit_from_net(res):
    max = -100
    max_i=0
    for i in range(10):
        if res[0][i] > max :
            max_i=i
            max=res[0][i]
    return max_i

def is_cont_frame(cont,box_size):
        x,y,w,h = cv2.boundingRect(cont)
        limit=box_size/4
        return (x<limit and y<limit) or (w>box_size*0.9) or (abs(box_size-y)< limit)
    

def get_digit_from_img(img,row,col,box_size,my_net):
        cur_img = img[row*box_size :(row+1)*box_size, col*box_size : (col+1)*box_size]
        contours ,hirer = cv2.findContours(cur_img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        max_cont= contours[0]     
        mask = np.zeros_like(cur_img)
        cv2.fillPoly(mask,[max_cont],1)
        cur_img=np.multiply(cur_img,mask)
        contours_left ,__= cv2.findContours(cur_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        max_cont_left = max(contours, key=cv2.contourArea)
        if (cv2.countNonZero(cur_img) == 0 or  is_cont_frame(max_cont_left,box_size)):
            return 0
        cur_img=cv2.resize(cur_img,(28 ,28))
        trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))])
        img_tensor = trans(cur_img)
        img_tensor= img_tensor.unsqueeze(0)
        res=(my_net(img_tensor))
        final_res = res.argmax(dim=1, keepdim=True)
        if (final_res is 0):
            return 9
        return  final_res

def img_to_board(img,my_net):
    board = np.zeros([9 , 9])
    mini_box_size = int (math.sqrt(np.size(img))/ 9 )
    for i in range(9):
        for j in range(9):
            board[i][j]=get_digit_from_img(img,i,j,mini_box_size,my_net)
    return board
            
def image_to_solve(img):
        learning_rate = 0.001
        momentum = 0.5
        network = Net()
        continued_optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate,momentum=momentum)
        network_state_dict = torch.load("model.pth")
        network.load_state_dict(network_state_dict)
        
        im=clean_img(img.copy())
        new_board=img_to_board(im,network)
        sudoko_solver.print_board(new_board)
        
def main():
        img = cv2.imread('images/sudo1.jpeg', 0)
        image_to_solve(img)




if __name__ == "__main__":
    main()
