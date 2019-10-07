import sys
import datetime
import time
from FileOp import *
#golbal variables
file = sys.argv[1]
dataList = []
visit = [False] * 81
flag = False

#function part
#check 
def check(pos, value):
    #row represennts the row of the nunbers
    #col of the numbers
    row = pos / 9
    col = pos % 9
    #check row
    for i in range(9):
        if dataList[row][i] == value:
            #print(1)
            return False
    #check col
    for j in range(9):
        if dataList[j][col] == value:
            #print(2)
            return False
    #check 3*3 the num located
    row = pos / 9 / 3 * 3
    col = pos % 9 / 3 * 3
    for i in range(row, row+3):
        for j in range(col, col+3):
            if dataList[i][j] == value:
               # print(2)
                return False
    return True

#the following func is for DFS

#build an arrry represent the grid that we visited
def dfs(pos):
    global flag
    if pos == 81:
        flag = True
        return
    row = pos / 9
    col = pos % 9
    #if grid now has non null value
    #then skip
    
    if dataList[row][col] != 0:
        visit[pos] = True
        dfs(pos+1)
    #choose num 1 to zero for now pos
    for i in range(1,10):
        if check(pos, i) and not visit[pos]:
            
            dataList[row][col] = i
            visit[pos] = True

            #go on
            dfs(pos+1)
            if flag:
                return
            #following part is backTracking
            dataList[row][col] = 0
            visit[pos] = False
            
#pragma mark - main

start_time = time.time()
readFile(file,dataList)
resultFile = getResultFile(file)
cal_start = time.time()
dfs(0)
cal_end = time.time()
writeResult1(resultFile,dataList)
end_time = time.time()

total_time(resultFile,round((end_time-start_time)*1000,3))
cal_time(resultFile,round((cal_end-cal_start)*1000,3))
