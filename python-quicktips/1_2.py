person_age = [
     ['john', 10],
     ['eva', 20],
     ['richard', 18],
     ['anabella', 23],
     ['david', 32]
]

# inplace sorting
person_age.sort(key=lambda x: x[1])
