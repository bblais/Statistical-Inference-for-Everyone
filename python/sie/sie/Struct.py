class Struct(dict):
    
    def __getattr__(self,name):
        
        try:
            val=self[name]
        except KeyError:
            val=super(Struct,self).__getattribute__(name)
            
        return val
    
    def __setattr__(self,name,val):
        
        self[name]=val


if __name__=="__main__":

    from Memory import *
    
    s=Struct({'hello':5})
    
    Remember(s)
    

