f = open("readme.md", 'r')
e = open("readme2.md", "w")

for g in f.readlines():
    if(g != "\n" and g[0] != '!'):
        l = "## "
        l += str(g[0]).upper()
        l += str(g[1:])
        e.write(l)
    
    else :
        e.write(g)

