import csv

with open('17-18_original_result.csv',newline='') as base_data:
    lines = csv.reader(base_data)# read in the csv data as line-iterator
    i = 1 #read header lock
    write_lines_b = [] # the output rows
    for line in lines:
        if i == 1: 
            i -= 1  # skip the headers
            continue
        else: 
            win_T = "" # win team
            lose_T = "" # lose team
            win_L = "" # win location('Home' or "Away")
            diff = 0 # score difference(former(visit) - latter(home))

            diff = int(line[3]) - int(line[5])
            if diff < 0: # home team wins
                win_T, lose_T = line[4].rstrip('*'), line[2].rstrip('*') # In case there is a '*' in the end of the string
                win_L = "H"
                diff = abs(diff)
            else: # away team wins
                win_T, lose_T = line[2].rstrip('*'), line[4].rstrip('*')
                win_L = "A"
                diff = abs(diff)
            write_lines_b.append([win_T, lose_T, win_L, diff])
                
with open('17-18_wl.csv','w') as output:
    newlines = csv.writer(output)    
    newlines.writerow(["win_T", "lose_T", "win_L", "diff"])# header
    newlines.writerows(write_lines_b)


with open('18-19_original_result.csv',newline='') as test_data:
    lines = csv.reader(test_data)# read in the csv data as line-iterator
    i = 1 #read header lock
    write_lines_t = [] # the output rows
    for line in lines:
        if i == 1: 
            i -= 1  # skip the headers
            continue
        else: 
            win_T = "" # win team
            lose_T = "" # lose team
            win_L = "" # win location('Home' or "Away")
            diff = 0 # score difference(former(visit) - latter(home))

            diff = int(line[3]) - int(line[5])
            if diff < 0: # home team wins
                win_T, lose_T = line[4].rstrip('*'), line[2].rstrip('*')
                win_L = "H"
                diff = abs(diff)
            else: # away team wins
                win_T, lose_T = line[2].rstrip('*'), line[4].rstrip('*')
                win_L = "A"
                diff = abs(diff)
            write_lines_t.append([win_T, lose_T, win_L, diff])
                
with open('18-19_wl.csv','w') as output:
    newlines = csv.writer(output)    
    newlines.writerow(["win_T", "lose_T", "win_L", "diff"])
    newlines.writerows(write_lines_t)