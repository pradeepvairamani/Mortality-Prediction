stopwordsfile = open('Stopwords.txt', 'r')

stopwordslist = []

for line in stopwordsfile:
    if len(line) > 1:
        stopwordslist.append(line.strip())


# print (stopwordslist)

infile = "newcleannotes.csv"
outfile = "cleannotes_nostopwords.csv"


def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]

fin = open(infile)
fout = open(outfile, "w+")
import csv
with open(infile, 'r', encoding="utf8") as csvfile, open('nostopwords.csv', 'w+', encoding="utf8") as outfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    spamwriter = csv.writer(outfile, delimiter=',')
    for row in spamreader:
        temp = row[2].split(" ")
        for word in stopwordslist:
            if word in temp:
                temp = remove_values_from_list(temp, word)
        temp1 = []
        for i, word in enumerate(temp):
            if len(word) > 1:
                temp1.append(word)
        # print(temp1)
        row[2] = " ".join(temp1)
        spamwriter.writerow(row)

# for line in fin:
#     for word in stopwordslist:
#         line = line.replace(word, "")
#         print line
#     fout.write(line)
fin.close()
fout.close()
