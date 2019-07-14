import csv
myData=[]
myText=[]

# csvFile=open('./data/saveCsv.csv','r')
# reader=csv.reader(csvFile)
# i=0
# for line in reader:
#     myData.append([eval(x) for x in line])
#     i+=1

csvFile=open('./data/saveText.csv','r')
reader=csv.reader(csvFile)
for line in reader:
    myText.append("".join(line))

print()

# for i in range(len(myText)):
#     print("Document",i)
#     for j in range(5):
#         print(myText[i][j+1])

# myFile=open('./data/sortText.txt','w')
# for row in myText:
#     print(row,file=myFile)