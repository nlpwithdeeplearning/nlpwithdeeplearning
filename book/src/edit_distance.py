def edit_distance(str0:str, str1: str) -> int:
    len0 = len(str0)
    len1 = len(str1)

    # initialize the table, table[len0][len1] is the final output
    table = [[0 for j in range(len1+1)] for i in range(len0+1)]
    
    for i in range(len0+1):
        for j in range(len1+1):
            # if str0 is empty, all of str1 needs to be inserted
            if i==0:
                table[i][j] = j
            # likewise for str1
            elif j==0:
                table[i][j] = i
            # if corresponding characters are matching, then go back diagonally
            elif str0[i-1] == str1[j-1]:
                table[i][j] = table[i-1][j-1]
            # this is either an insertion (move right) or deletion (up) or swap (diagonal) then based on which path is shorter
            else:
                table[i][j] = 1 + min( # 1 -> equal cost assumed
                    table[i][j-1], # right arrow = insert
                    table[i-1][j], # up arrow = delete
                    table[i-1][j-1], # diagonal = replace
                )
    
    return table[len0][len1]

if __name__ == '__main__':
    str0 = 'hare'
    str1 = 'hari'
    ed = edit_distance(str0, str1)
    print(ed)