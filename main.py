import numpy as np
import torch 
import torch.nn.functional as F
import torchvision
import cv2
import math



class sudoNN(torch.nn.Module):
        def __init__(self):
            super(sudoNN, self).__init__()
            self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
            self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = torch.nn.Dropout2d(0.25)
            self.dropout2 = torch.nn.Dropout2d(0.5)
            self.fc1 = torch.nn.Linear(9216, 128)
            self.fc2 = torch.nn.Linear(128, 10)

        def forward(self,x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output
    
def textuali():
    my_list=[]
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    fonts = [ cv2.FONT_HERSHEY_SIMPLEX , cv2.FONT_HERSHEY_TRIPLEX ,cv2.FONT_ITALIC , cv2.FONT_HERSHEY_SCRIPT_COMPLEX]
    for font in fonts:
        for i in range(10):
            img = np.zeros((100,100),dtype='uint8')
            cv2.putText(img,  str(i),(20,70), font, 2,(255),5,cv2.LINE_AA)
            img=cv2.resize(img,(28 ,28))
            img_tensor = transform(torchvision.transforms.ToPILImage(mode='L')(img))
            img_tensor_array = img_tensor.unsqueeze(0)
            my_list.append([img_tensor_array,i])
            cv2.imshow('hry',img)
            print(img)
            cv2.waitKey(0)
    return my_list   

def train_net():
        my_batch_size = 32
        num_epochs = 10
        learn_rate = 0.0001
        
        train_db = torchvision.datasets.MNIST(root ="./MNIST",train=True ,transform=torchvision.transforms.ToTensor(),download=True)
        test_db = torchvision.datasets.MNIST(root='./MNIST', train=False ,transform=torchvision.transforms.ToTensor(), download=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_db, batch_size=my_batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_db,  batch_size=my_batch_size, shuffle=False)
        
        
        my_model = sudoNN()
        optimizer = torch.optim.Adam(my_model.parameters(),lr=learn_rate)
        total_step = len(train_loader)
        loss=0
                
        my_list= textuali()
        for epoch in range(num_epochs):
            for i, (images,labels) in enumerate(my_list):
                out = my_model(images)
                loss = F.nll_loss(out, torch.tensor([labels]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        for epoch in range(num_epochs):
            for i, (images,labels) in enumerate(train_loader):
                out = my_model(images)
                loss = F.nll_loss(out,labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        torch.save(my_model, './my_mnist_model.pt') 
        print("loss is:")
        print(loss*100)
        return my_model


        
        
def print_board(board):
        for i in range(len(board)):
            if (i%3 == 0 ):
                print ("-----------------------")
            for j in range(len(board[0])):
                if (j%3 ==0 and j!=0) :
                    print (" | " ,end="")
                print(int(board[i][j]) , end=" ")
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
        limit=box_size/2
        return (x<limit/5 and y<limit/5) or (w>box_size*0.9) 

def thin_num(img):
        kernel = np.ones((2,2),np.uint8)
        im_bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        erosion = cv2.erode(im_bw,kernel,iterations = 1)
        return erosion
    
def get_digit_from_img(img,row,col,box_size,my_net):
        cur_img = img[row*box_size :(row+1)*box_size, col*box_size : (col+1)*box_size]
        contours ,hirer = cv2.findContours(cur_img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        max_cont= contours[0]
        mask = np.zeros_like(cur_img)
        cv2.fillPoly(mask,[max_cont],1)
        cur_img=np.multiply(cur_img,mask)
        if ( cv2.countNonZero(cur_img) ==0 or is_cont_frame(max_cont,box_size)):
            return 0
        cur_img=cv2.resize(cur_img,(28 ,28))
        print(cur_img)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        img_tensor = transform(torchvision.transforms.ToPILImage(mode='L')(cur_img))
        img_tensor_array = img_tensor.unsqueeze(0)
        res=(my_net(img_tensor_array))
        return  res.argmax(dim=1, keepdim=True) 

def img_to_board(img,my_net):
    board = np.zeros([9 , 9])
    mini_box_size = int (math.sqrt(np.size(img))/ 9 )
    for i in range(9):
        for j in range(9):
            board[i][j]=get_digit_from_img(img,i,j,mini_box_size,my_net)
    return board
            
        


    
def img_proccess(img,my_net):
        im=clean_img(img.copy())
        new_board=img_to_board(im,my_net)
        print_board(new_board)
        
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
        #my_net=train_net()
        nodel = torch.load("./my_mnist_model.pt")
        nodel.eval()
        img = cv2.imread('images/sudo1.jpeg', 0)
        img_proccess(img,nodel)


        



if __name__ == "__main__":
    main()
