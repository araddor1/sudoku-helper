


# printing the board function 
# input : board array
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

# finding the first empty spot on board 
# input : board . returns output in next arr  
def find_empty_spot(board,next):
        for i in range(len(board)):
            for j in range(len(board)):
                if(board[i][j]==0):
                    next[0]=i
                    next[1]=j
                    return True
        return False

# returns if locating num in row num is legal
def legal_in_row(board,row_num,num):
        for i in range(len(board)):
            if(board[row_num][i]==num):
                    return False
        return True

# returns if locating num in col num is legal
def legal_in_col(board,col_num,num):
        for i in range(len(board)):
            if(board[i][col_num]==num):
                    return False
        return True

# returns if num is legal in his box
def legal_in_box(board,row,col,num):
        sqrt_len=int(math.sqrt(len(board)))
        base_row = row - row%sqrt_len
        base_col = col- col%sqrt_len
        for i in range(sqrt_len):
            for j in range(sqrt_len):
                if(board[base_row+i][base_col+j]==num):
                    return False
        return True           

# returns if num in row,col on board is legal
def legal_pos(board,row,col,num):
        return legal_in_col(board,col,num) and legal_in_row(board,row,num) and legal_in_box(board,row,col,num)

#  Solves the sudoko 
# returns solvd sudoko in board. returns false if no possible option
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