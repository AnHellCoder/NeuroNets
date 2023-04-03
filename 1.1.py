am = {1: [4, 8], 2: [3, 6, 8], 3: [2], 4: [1, 5, 7], 5: [4, 6, 7], 6: [2, 5, 7], 7: [4, 5, 6, 8], 8: [1, 2, 7]}
g = {'А': ['Б', 'В', 'Г'], 'Б': ['А', 'Г', 'Д'], 'В': ['А', 'Е'], 'Г': ['А', 'Б', 'Д', 'Е'], 'Д': ['Б', 'Г', 'К'], 'Е': ['В', 'Г', 'К'], 'К': ['Д', 'Е', 'Л'], 'Л': ['К']}
full = {}

lstF = [False for i in range(1, 9)]
lstM = [i for i in range(1, 9)]
lstG = ['А','Б','В','Г','Д','Е','К','Л']

for i in range(len(lstM)):
    for j in range(len(am[lstM[i]])):
        v = am[lstM[i]][j]
        am[lstM[i]][j] = len(am[v])
    
    am[lstM[i]].sort()

for i in range(len(lstG)):
    for j in range(len(g[lstG[i]])):
        v = g[lstG[i]][j]
        g[lstG[i]][j] = len(g[v])
    
    g[lstG[i]].sort()

for i in range(len(lstM)):
    for j in range(len(lstG)):
        if(am[lstM[i]] == g[lstG[j]] and not lstF[j]):
            full[lstM[i]] = lstG[j]
            lstF[j] = True
            break

for i in range(len(lstM)):
    print('{0}: {1}'.format(lstM[i], full[lstM[i]]))


#############################################

at = {1: [2, 4], 2: [1, 4, 6], 3: [4, 5], 4: [1, 2, 3, 5, 6], 5: [3, 4], 6: [2, 4]}
g = {'a': ['b', 'g'], 'b': ['a', 'c', 'd', 'e', 'g'], 'c': ['b', 'd'], 'd': ['b', 'c', 'e'], 'e': ['b', 'd'], 'g': ['a', 'b']}
full = {}

lstm = [i for i in range(1, 7)]
lstg = ['a', 'b', 'c', 'd','e','g']

for i in range(len(lstm)):
    for j in range(len(at[lstm[i]])):
        v = at[lstm[i]][j]
        at[lstm[i]][j] = len(at[v])
    
    at[lstm[i]].sort()

for i in range(len(lstg)):
    for j in range(len(g[lstg[i]])):
        v = g[lstg[i]][j]
        g[lstg[i]][j] = len(g[v])
    
    g[lstg[i]].sort()

for i in range(len(lstm)):
    for j in range(len(lstg)):
        if(at[lstm[i]] == g[lstg[j]]):
            full[lstm[i]] = lstg[j]

for i in range(len(lstm)):
    print('{0}: {1}'.format(lstm[i], full[lstm[i]]))