
def determine_label((x,y,z)):
    if  x > 3:
        if y > 1:
            return 1
        elif y <= 1 and y > 0:
            return 2
        elif y <= 0 and y > -1:
            return 3
        elif y <= -1:
            return 4
    elif x <= 3 and x > 2:
        if y > 1:
            return 5
        elif y <= 1 and y > 0:
            return 6
        elif y <= 0 and y > -1:
            return 7
        elif y <= -1:
            return 8
    elif x <= 2:
        if y > 1:
            return 9
        elif y <= 1 and y > 0:
            return 10
        elif y <= 0 and y > -1:
            return 11
        elif y <= -1:
            return 12