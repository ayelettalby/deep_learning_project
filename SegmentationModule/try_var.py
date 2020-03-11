cars=['a','b','c','d','e']

class car():
    def __init__(self,col,num,size,letter):
        self.color=col
        self.num=num
        self.size=size
        self.letter=letter
        self.letter_a='a'
        self.letter_b = 'b'
        self.letter_c = 'c'

    def __repr__(self):
        return (self.letter_a)

    def forward(self):
        let='letter_'+self.letter
        a=self.let
        return a


new_car=car(1,2,3,'a')
b=new_car.forward()
print (new_car)
