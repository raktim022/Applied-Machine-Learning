import score
import pickle
import numpy
import os
import requests
import subprocess
import time
import unittest


model_loc="Applied-Machine-Learning/Assignments/models/lgr.pkl"
model=pickle.load(open(model_loc,"rb"))

sent="I like cats"
threshold=0.7

label,prop=score.score(sent,model,threshold)

class TestFunction:
    def smoke_test(self):
         
        assert label!= None
        assert prop!= None
    
    def test_format(self):
        assert type(sent) == str
        assert type(threshold) == float 
        assert type(label) == numpy.int64
        assert type(prop) == numpy.float64 

    def test_pred(self):
        assert label ==0 or label ==1

    def test_propensity(self):
        assert prop>=0 and prop<=1

    def prop_test_0(self):
        label,prop=score.score(sent,model,0)
        assert label == 1

    def prop_test_1(self):
        label,prop=score.score(sent,model,1)
        assert label == 0

    def test_spam(self):
        label,prop=score.score("YOU HAVE WON 1 MILLION DOLLARS. SEND YOUR ACCOUNT DETAILS!",model,threshold)
        assert label == 1


    def test_spam(self):
        label,prop=score.score("I like cats.",model,threshold)
        assert label == 0





class TestFlask(unittest.TestCase):
    def test_flask(self):
        # Launch the Flask app using os.system
        os.system('python app.py &')

        # Wait for the app to start up
        time.sleep(1)

        # Make a request to the endpoint
        response = requests.get('http://127.0.0.1:5000/')
        print(response.status_code)

        # Assert that the response is what we expect
        self.assertEqual(response.status_code, 200)
        print("OK")
        self.assertEqual(type(response.text), str)
        print("OKAY")

        # Shut down the Flask app using os.system
        os.system('kill $(lsof -t -i:5000)')


if __name__ == '__main__':
    unittest.main()


