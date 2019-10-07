import os
import time
import sys
from FileOp import * 
file = sys.argv[1]

def cross(A,B):
    return [a+b for a in A for b in B]

digits = '123456789'
rows_num = digits
cols_num = '123456789'
sudoku_pos = cross(rows_num,cols_num)
# dataList = []
#unitlist has three parts
#all rows
#all cols
#all 3*3 grids
three_check_part = [[row+col for row in rows_num] for col in cols_num] + [[row+col for col in cols_num] for row in rows_num] + [[row+col for row in rows for col in cols] for rows in ('123','456','789') for cols in ('123','456','789')]

#units
#key is every 81 pos
#value is the row col grid the pos belonged to
total_Pos_check = dict((index, [check_space for check_space in three_check_part if index in check_space])for index in sudoku_pos)

#every pos has 20 peers 9+9-2+9-5
single_Pos_check = dict((index, (set(total_Pos_check[index][0] + total_Pos_check[index][1] + total_Pos_check[index][2]) - set([index]))) for index in sudoku_pos)


     
def get_value(dataList):
	#every pos has nine prob choice
    values = dict((s, digits) for s in sudoku_pos)

    sudoku_dict = dict(zip(sudoku_pos, dataList))

    #CHOOSE posible value
    for key,value in sudoku_dict.items(): 
        if value in digits:

        	#according to related value 
        	#tick imposible value
            if not tick(values, key, value): 
                return False
    return values
#tick the posibility of value
def tick(values, key, value):
	#the rest are all possible values
    nov_values = values[key].replace(value, '')
    if all(eliminate(values, key, value2) for value2 in nov_values):
        return values
    else:
        return False


#remove related value in key
#update related relations of its three parts
def eliminate(values, key, value): 
    if value not in values[key]:
        return values 

    #remove value
    values[key] = values[key].replace(value,'')

    #if after remove
    #no value left, it's wrong
    if len(values[key]) == 0: 
        return False

    # if left one value
    #remove it from it's three parts
    elif len(values[key]) == 1: 
        v = values[key]
        if not all(eliminate(values, k, v) for k in single_Pos_check[key]):
            return False
    
    #search
    #if find only one pos can hold the value
    #put it in
    for check in total_Pos_check[key]: 
        pos = [k for k in check if value in values[k]]
        if len(pos) == 0:
            return False
        elif len(pos) == 1 and tick(values, pos[0], value) == False: 
            return False
    return values

#this is the key function
#every time we choose values that there are little possibility
def search(values): 
    if not values:
        return False
    #find the result, just return 
    if all(len(values[pos]) == 1 for pos in sudoku_pos):
    	return values
    #choose the little possibility key
    next_length,next_key = min((len(values[pos]), pos) for pos in sudoku_pos if len(values[pos]) > 1) 
    
    #recursive , decrease the possibility to find the result
    for result in (search(tick(values.copy(), next_key, v)) for v in values[next_key]): 
        if result:
            return result
    return False

resultFile = getMRVResultFile(file)
start_time = time.time()
dataList = readData(file)
cal_start = time.time()
result = search(get_value(dataList))
cal_end = time.time()
writeResult(resultFile,result)
end_time = time.time()

total_time(resultFile,round((end_time-start_time)*1000,3))
cal_time(resultFile,round((cal_end-cal_start)*1000,3))

