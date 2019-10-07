#file operate function part
########################################
#######      CSP
def readFile(fileName,dataList):
    f = open(fileName, 'r')
    while True:
        line = f.readline()
        if line:
            data = []
            for i in line.split(' '):
                data.append(int(i))
            dataList.append(data)
        else:
            break
   
def getResultFile(fileName):
    s = fileName
    result = []
    for i in s.split('.'):
        result.append(i)
    return result[0]+'-CSP-Result.txt'


def writeResult1(file,dataList):
    f = open(file,'w')
    for i in range(9):
        for j in range(9):
            f.write(str(dataList[i][j])+' ')
            if(j+1)%3 == 0:
                f.write(' ')
        f.write('\n')
        if(i+1)%3 == 0:
            f.write('\n')
    f.close()


########################################
#######      MRV
digits = '123456789'
rows_num = digits
cols_num = '123456789'
def readData(fileName):
    dataList = []
    f = open(fileName, 'r')
    while True:
        line = f.readline()
        if line:
            data = []
            for i in line.split():
                dataList.append(i)      
        else:
            break

    return dataList 

def getMRVResultFile(fileName):
    s = fileName
    result = []
    for i in s.split('.'):
        result.append(i)
    return result[0]+'-MRV-Result.txt'

def writeResult(fileName,result):
    f = open(fileName, 'w')
    for row in rows_num:
        for col in cols_num:
            f.write(result[row+col]+' ')
            if(int(col) % 3 == 0):
                f.write(' ')
        f.write('\n')
        if(int(row) % 3 == 0):
            f.write('\n')

#########
def total_time(file,cost):
    f = open(file,'a')
    s = 'program total running time: '+str(cost)+' ms\n'
    f.write(s)
    f.close()

def cal_time(file,cost):
    f = open(file,'a')
    s = 'program algorithm calculating time: '+str(cost)+' ms\n'
    f.write(s)
    f.close()
