import os

config = {}

if os.path.exists(".env"):
	with open(".env", "r") as f:
		lines = f.readlines()
		for line in lines:
			equals_index = line.index("=")
			key = line[:equals_index]
			value = line[equals_index+1:].strip()
			config[key] = value