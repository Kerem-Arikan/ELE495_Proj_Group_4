def label2key():
    keys = open("keymap.txt", "r")
    gestures = open("../saved_model/labelmap.txt","r")
    mydic={}

    while True:
        dumbkey = keys.readline()
        dumbgesture = gestures.readline()
        if dumbkey == '':
            break
        if dumbgesture == '':
            break
        if dumbkey[-1]=="\n":
            dumbkey = dumbkey[:-1]
        if dumbgesture[-1]=='\n':
            dumbgesture = dumbgesture[:-1]
        mydic[dumbkey]=dumbgesture

    keys.close()
    gestures.close()

    return mydic
