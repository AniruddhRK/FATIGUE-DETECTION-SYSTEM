def checkBlinkStatus(eyeStatus,state,blinkCount,drowsy,falseBlinkLimit,drowsyLimit):
    if(state >= 0 and state <= falseBlinkLimit):
        if(eyeStatus):
            state = 0
            

        else:
            state += 1

    elif(state >= falseBlinkLimit and state < drowsyLimit):
        if(eyeStatus):
            blinkCount += 1 
            state = 0
            

        else:
            state += 1


    else:
        if(eyeStatus):
            state = 0
            drowsy = 1
            blinkCount += 1

        else:
            drowsy = 1

    return blinkCount,state,drowsy