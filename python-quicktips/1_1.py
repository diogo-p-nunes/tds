person_age = {
	'john': 10,
	'eva' : 20,
	'richard': 18,
	'anabella': 23,
	'david': 32
}

sorted_person_age = dict(sorted(person_age.items(), key = lambda item: item[1]))
