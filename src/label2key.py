class label2key(object):
	"""docstring for label2key"""
		
	def label2key(keys_path="keymap.txt", gestures_path="../frozen_graph/labelmap.txt"):
		dic_data={}
		try:
			keys = open(keys_path, "r")
			gestures = open(gestures_path,"r")
			while True:
				dumbkey = keys.readline()
				dumbgesture = gestures.readline()
				# End of file check
				if dumbkey == '':
					break
				if dumbgesture == '':
					break
				# Check if there is new line symbol.
				if dumbkey[-1]=="\n":
					dumbkey = dumbkey[:-1]
				if dumbgesture[-1]=='\n':
					dumbgesture = dumbgesture[:-1]
				
				dic_data[dumbgesture]=dumbkey
			keys.close()
			gestures.close()	
		except IOError:
			print("IOError")
			print("gestures_path="+gestures_path)
			print("keys_path="+keys_path)
		return dic_data


		
