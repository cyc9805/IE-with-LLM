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


denoising_task_prompt = '''
You are tasked with predicting and reconstructing missing segments in a given text sequence. The input will have parts replaced by mask tokens like {0}, {1}, etc. Your goal is to fill in these placeholders with contextually appropriate text.

Input and Output Example:

The dialogue input:
Mr. Dursley was the director of a firm called {0}, which made {1}. He was a big, solid man with a bald head. Mrs. Dursley was thin and {2} neck, which came in very useful as she spent so much of her time {3}.

Output:
{{"{0}": "Grunnings", "{1}": "drills", "{2}": "had a long", "{3}": "craning over garden fences"}}

Guidelines:

	1.	Contextual Accuracy: Use surrounding text to infer the appropriate content for each amsk token.
	2.	Coherence: Ensure the sequence flows naturally and maintains the narrative’s integrity.
	3.	Detail Preservation: Incorporate details that align with the text’s tone and style.
	4.	Avoid Repetition: Prevent unnecessary repetition unless contextually justified.

Follow these guidelines to accurately predict and fill in the missing segments for any given input sequence.
Output the final structure as a python dictionary.
Do not provide any explanation for your answer.
'''