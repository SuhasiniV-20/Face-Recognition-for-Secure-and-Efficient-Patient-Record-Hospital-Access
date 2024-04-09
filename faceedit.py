from flask import Flask, render_template, request, redirect, url_for,jsonify
import cv2
import os
import json
import numpy as np
import csv
import pandas as pd
from PIL import Image, ImageFont, ImageDraw
import datetime
from flask_cors import CORS
import time
import sqlite3

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
REGISTER_PATH="./Register/"
TEMP_PATH="./Capturing_Images"
upload_path='./upload_face'
upload_path2='./upload_face2'
app.config['SEND_FILE_MAX_AGE_DEFAULT']=1
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

conn = sqlite3.connect('db.sqlite3')
c = conn.cursor()
#alter_query = '''ALTER TABLE users 
                 #ADD COLUMN ph TEXT'''
#c.execute(alter_query)
c.execute('''CREATE TABLE IF NOT EXISTS srm (
                name TEXT,
                Id TEXT,
                Gender TEXT,
                phonenumber TEXT,
                address TEXT,
                Email TEXT
            )''')

###c.execute('''(DESC users)''')
conn.commit()
conn.close()

@app.route('/patientreg', methods=['GET', 'POST'])

def upload_register():
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return json.dumps({"status": "Error", "msg": "Image cannot be empty "})
        

    return "Registration successful!"
       
       
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

@app.route('/')
def index():
    return render_template('frontpage.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')
    print(email.lower())
    print(password)
    
                        
    if (email.lower() == 'srm@gmail.com' and password == 'srm'):
        return json.dumps ({"status": "true","message": "User Registered Succesfully"})
    else:
        return json.dumps ({"status": "false","message": "User Not Registered"})

        
    

@app.route('/take_image', methods=['GET', 'POST'])
def TakeImages():
    name = request.form.get('name')
    print(name)
    Id = request.form.get('Id')
    Gender = request.form.get('gender')
    phonenumber = request.form.get('phonenumber')
    address = request.form.get('address')
    Email = request.form.get('email')
    print(Id)
    print(Gender)
    print(phonenumber)
    print(address)
    print(Email)
    
    conn = sqlite3.connect('db.sqlite3')
    c = conn.cursor()
        #c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, password))
    c.execute("INSERT INTO srm (name, Id, Gender, phonenumber, address, Email) VALUES (?, ?, ?, ?, ?, ?)",
              (name, Id, Gender, phonenumber, address, Email))
    conn.commit()
    conn.close()
    
    
    #Id='4123'
    #name='ghhwewewe'
    age='34'
    #gender='m'
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        print("inside if cond")
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                #incrementing sample number 
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("Capturing_Images\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                cv2.imshow('frame',img)
            #wait for 100 miliseconds 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum>60:
                break
        
        cam.release()
        cv2.destroyAllWindows() 
        res = "Images Saved for ID : " + Id +" Name : "+ name +"Age :" + age + "Gender:" +Gender
        row = [Id , name,age,Gender]
        with open('patient_List\patient_List.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        
        #message.configure(text= res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)
    recognizer = cv2.face_LBPHFaceRecognizer.create()#recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("Capturing_Images")
    recognizer.train(faces, np.array(Id))
    recognizer.save("Models\Trainner.yml")
   # return json.dumps ({"status": "true","message": "User Registered Succesfully"})
    #res = "Image Trained"#+",".join(str(f) for f in Id)
    #message.configure(text= res)


def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #print(imagePaths)
    #print(imagePaths)
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids

# ... (previous code)
@app.route('/track_images', methods=['GET', 'POST'])
#@app.route('/track_images')
def TrackImages():
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("Models\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("patient_List\patient_List.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time','Location']
    attendance = pd.DataFrame(columns = col_names)   
    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            print(conf)
            if(conf < 50):
                Location="College"
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa+"-"+"Student"
               
              
                
                name=""
                phonenumber=""
                gender=""
                address=""
                email=""
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp,Location]
                #break
                con = sqlite3.connect('db.sqlite3')
                #Id =4123
    
                #completion = False
                with con:
                    cur = con.cursor()
                    cur.execute("SELECT * FROM srm WHERE Id = ?",(Id,))
                    #cur.execute("SELECT * FROM srm WHERE Id == :Id", {"Id": str(Id)})
                    rows = cur.fetchone()

                    #for row in rows:
                    if rows:
                        print(rows)                        
                        name = rows[0]
                        gender = rows[2]
                        phonenumber = rows[3]
                        address = rows[4]
                        email = rows[5]
                    else:
                        print("rows not found")
                    #con.commit()
                    #row = cur.fetchone()
                    #name = row[1]
                    #gender = row[2]
                    #phonenuber = row[3]
                    #address = row[4]
                    #email = row[5]
                #c.execute("SELECT * FROM srm WHERE Id=?",(Id))
                return json.dumps ({"status": "true","message": "face Detected","Id": str(Id),"name": name,"gender": gender,"phonenumber": phonenumber,"address": address,"email": email})
                
               
                        
               
            else:
                Id='Not_MATCHED'                
                tt=str(Id)  
            
            if(conf > 75):
                noOfFile=len(os.listdir("Database"))+1
                cv2.imwrite("Database\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])  
            
            
            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')
        
        cv2.imshow('Face_Recognize',im) 
        
        if ((cv2.waitKey(1)==ord('q'))):
            break
   
    return json.dumps ({"status": "false","message": "No face Detected"})
    #return redirect(url_for('result', message=message))






@app.route('/result/<message>')
def result(message):
    return render_template('result.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)
