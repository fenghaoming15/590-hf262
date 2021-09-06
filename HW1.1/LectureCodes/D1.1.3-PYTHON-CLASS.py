#FOR MORE: https://docs.python.org/3/tutorial/classes.html

#REMEMBER 
#AND OBJECT IS A SPECIFIC INSTANCE OF A CLASS
#THE CLASS ITSELF IS A TEMPLATE FOR OBJECTS 
class Dog:

    # class variable shared by all instances
    kind = 'canine'        

    #INITIALIZE
    def __init__(self, attributions):
        self.name = attributions[0]      # instance variable unique to each instance
        self.weight = attributions[1]    
        self.possesions=[]               #initial as empty, fill later

    def increase_weight(self,dw=1):
    	self.weight+=dw


#INITIALIZE TWO OBJECTS OF CLASS DOG
L = Dog(['Luna' ,40])
S = Dog(['Spark',50])


#SEE INITIAL ATTRIBUTES
print("#-----------------------")
print(L.name, L.weight, L.kind)
print(S.name, S.weight, S.kind)

#RUN THE increase_weight() METHOD
print("#-----------------------")
L.increase_weight()
print(L.name, L.weight, L.kind)
L.increase_weight(5)
print(L.name, L.weight, L.kind)


#POPULATE POSSSESSION
print("#-----------------------")
print(S.name, S.weight, S.kind,S.possesions)
S.possesions=['collar','leash','bowl']
print(S.name, S.weight, S.kind,S.possesions)
S.possesions.append('dog food')
print(S.name, S.weight, S.kind,S.possesions)


#SUBCLASS
    #NOTICE IT INHERITS ATTRIBUTIONS OF CLASS Dog
class SmallDog(Dog):
    size="small"
    
    # provides new attributions 
    # but does not break __init__()
    def update(self, H):
        self.height=H

B = SmallDog(['Bo' ,15])
B.update(1)
print(B.name, B.weight, B.kind, B.possesions,B.size,"H=",B.height)
