class label2key(object):
	"""docstring for label2key"""
	def __init__(self, keys_path="keymap.txt", gestures_path="../frozen_graph/labelmap.txt"):
		super(label2key, self).__init__()
		label2key.__data={}
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
				
				label2key.__data[dumbgesture]=dumbkey
			keys.close()
			gestures.close()	
		except IOError:
			print("IOError")
			print("gestures_path="+gestures_path)
			print("keys_path="+keys_path)

	def label2key(self,LABEL):
		GESTURE=""
		try:
			GESTURE=self.__data[LABEL]
		except:
			return "unknown"
		return GESTURE

		
