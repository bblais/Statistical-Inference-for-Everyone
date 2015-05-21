def combinations(items):

    if not items:
        yield []
    else:
        for i,item in enumerate(items):
            new_items=items[:]
            s=new_items.pop(i)
            for c in combinations(new_items):
                y=[item]
                y.extend(c)
                yield y


s=set()

for v in combinations(['A','A','A','D','D']): 
    print v
    s.add(tuple(v))
    
for i,item in enumerate(s):
    print i+1,'&',' '.join(item),r'\\'