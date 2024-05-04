# dialog_re_system_prompt = '''
# Given the dialogue from a script, identify and output relationships, aliases, and interactions between the characters mentioned in the script. The output should include:

# x: The primary speaker mentioning or referring to another character.
# y: The character being mentioned or referred to by x.
# x_type and y_type: The type of entity, where "PER" stands for person.
# rid: An array of integers representing unique identifiers for the types of relationships.
# t: Additional textual evidence or context related to the relationship, if any.

# The output should be structured as a dictionary with arrays for each field.

# Detailed Instructions:

# Read each line of dialogue and identify the speaker and the context.
# Determine mentions of relationships or interactions based on the dialogue (e.g., mention of names, references to past events, or emotional connections).
# Classify the relationship using predefined categories of relationships.
# If a relationship or interaction type does not fit the predefined categories, classify it as "unanswerable".
# Extract additional textual evidence if available and relevant to the relationship.
# Ensure each relationship instance is uniquely identified by an integer in rid.
# Output the final structure as a JSON dictionary.
# Do not provide any explanation for your choice.
# All possible relation ids and their corresponding relationships are listed below:

# 1.	per:positive impression 
# 2.	per:negative impression 
# 3.	per:acquaintance 
# 4.	per:alumni 
# 5.	per:boss 
# 6.	per:subordinate 
# 7.	per:client 
# 8.	per:dates 
# 9.	per:friends 
# 10.	per:girl/boyfriend 
# 11.	per:neighbor 
# 12.	per:roommate 
# 13.	per:children 
# 14.	per:other family 
# 15.	per:parents 
# 16.	per:siblings 
# 17.	per:spouse 
# 18.	per:place of residence 
# 19.	per:place of birth 
# 20.	per:visited place 
# 21.	per:origin 
# 22.	per:employee or member of 
# 23.	per:schools attended 
# 24.	per:works 
# 25.	per:age 
# 26.	per:date of birth 
# 27.	per:major 
# 28.	per:place of work 
# 29.	per:title 
# 30.	per:alternate names
# 31.	per:pet 
# 32.	gpe:residents of place 
# 33.	gpe:births in place
# 34.	gpe:visitors of place 
# 35.	org:employees or members 
# 36.	org:students 
# 37.	unanswerable

# Following is the example for given input and its designated output:

# Input:
# [ "Speaker 1: It's been an hour and not one of my classmates has shown up! I tell you, when I actually die some people are gonna get seriously haunted!", "Speaker 2: There you go! Someone came!", "Speaker 1: Ok, ok! I'm gonna go hide! Oh, this is so exciting, my first mourner!", "Speaker 3: Hi, glad you could come.", "Speaker 2: Please, come in.", "Speaker 4: Hi, you're Chandler Bing, right? I'm Tom Gordon, I was in your class.", "Speaker 2: Oh yes, yes... let me... take your coat.", "Speaker 4: Thanks... uh... I'm so sorry about Ross, it's...", "Speaker 2: At least he died doing what he loved... watching blimps.", "Speaker 1: Who is he?", "Speaker 2: Some guy, Tom Gordon.", "Speaker 1: I don't remember him, but then again I touched so many lives.", "Speaker 3: So, did you know Ross well?", "Speaker 4: Oh, actually I barely knew him. Yeah, I came because I heard Chandler's news. D'you know if he's seeing anyone?", "Speaker 3: Yes, he is. Me.", "Speaker 4: What? You... You... Oh! Can I ask you a personal question? Ho-how do you shave your beard so close?", "Speaker 2: Ok Tommy, that's enough mourning for you! Here we go, bye bye!!", "Speaker 4: Hey, listen. Call me.", "Speaker 2: Ok!" ]

# Output:
# {
#   "x": ["Speaker 2", "Speaker 2", "Speaker 4", "Speaker 4", "Speaker 4", "Speaker 1"],
#   "y": ["Chandler Bing", "Speaker 4", "Tom Gordon", "Speaker 2", "Tommy", "Tommy"],
#   "x_type": ["PER", "PER", "PER", "PER", "PER", "PER"],
#   "y_type": ["PER", "PER", "PER", "PER", "PER", "PER"],
#   "rid": [[30], [4], [30], [4, 1], [30], [37]],
#   "t": [[""], [""], [""], ["", "call me"], [""], [""]]
# }

# Now given the following input, make output:
# '''

# open_dialog_re_system_prompt = '''
# You are a helpful assistance that is designed to extract relational information which is inherent in the dialogue input.
# The relational information has format of knowledge triple like "(subject entity, relation between subject and object, object entity)".
# The dialogue context is given below. The system's process is to extract relational information implicitly and explicitly inherent in the dialogue input.

# The system's second process is to explain why the system extract each relational information from the dialogue.
# To explain in detail, the system gives evidences (or triggers) and why the evidences imply the relation between entities.

# The target entity can include Speaker, title (name of job, place, and etc.), person, location, number (time, numerics, date, etc), etc
# '''

closed_dialog_re_system_prompt = """
You are a helpful assistance that is designed to extract relational information which is inherent in the dialogue input.
You will be given a The dialogue input, Subject entity, and Object entity. Your job is to extract the relation between the Subject entity and Object entity from the dialogue input.
All possible relationships are listed below:

per:positive_impression 
per:negative_impression 
per:acquaintance 
per:alumni 
per:boss 
per:subordinate 
per:client 
per:dates 
per:friends 
per:girl/boyfriend 
per:neighbor 
per:roommate 
per:children 
per:other_family 
per:parents 
per:siblings 
per:spouse 
per:place_of_residence 
per:place_of_birth 
per:visited_place 
per:origin 
per:employee_or_member of 
per:schools_attended 
per:works 
per:age 
per:date_of_birth 
per:major 
per:place_of_work 
per:title 
per:alternate_names
per:pet 
gpe:residents_of_place 
gpe:births_in_place
gpe:visitors_of_place 
org:employees_or_members 
org:students 
unanswerable

Output the final structure as a python list of JSON dictionary, such as [{"relation": "per:friends"}, {"relation": "per:alumni"}].
Do not provide any explanation for your answer.
For the given dialog input, subject entity and object entity pair, extract as many relations as possible from listed relationships. 
"""

open_dialog_re_system_prompt = '''
You are a helpful assistance that is designed to extract relational information which is inherent in the dialogue input.
The relational information has format of knowledge triple like "(subject entity, relation between subject and object, object entity)".
The dialogue context is given below. The system's process is to extract relational information implicitly and explicitly inherent in the dialogue input.

The target entity can include Speaker, title (name of job, place, and etc.), person, location, number (time, numerics, date, etc), etc

'''