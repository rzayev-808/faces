from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, HttpResponse, redirect
from django.contrib import messages
import bcrypt
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import cv2
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from .serializers import FileSerializer
from django.contrib.auth import logout
import numpy as np
from ssl import SSLContext,PROTOCOL_TLSv1
from urllib.request import urlopen
from django.db.models import Q
from main.models import User, Person, ThiefLocation


class FileView(APIView):
  parser_classes = (MultiPartParser, FormParser)
  def post(self, request, *args, **kwargs):
    file_serializer = FileSerializer(data=request.data)
    if file_serializer.is_valid():
      file_serializer.save()
      return Response(file_serializer.data, status=status.HTTP_201_CREATED)
    else:
      return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)



def index(request):
    return render(request, 'session/login.html')

def addUser(request):
    return render(request, 'home/add_user.html')

def addCitizen(request):
   return render(request, 'home/add_citizen.html')

def logout_view(request):
    logout(request)
    messages.add_message(request,messages.INFO,"Successfully logged out")
    return redirect(index)


def viewUsers(request):
    users = User.objects.all()
    context = {
        "users": users
    }
    return render(request, 'home/view_users.html',context)

def saveUser(request):
    errors = User.objects.validator(request.POST)
    if len(errors):
        for tag, error in errors.iteritems():
            messages.error(request, error, extra_tags=tag)
        return redirect(addUser)

    hashed_password = bcrypt.hashpw(request.POST['password'].encode(), bcrypt.gensalt())
    user = User.objects.create(
        first_name=request.POST['first_name'],
        last_name=request.POST['last_name'],
        email=request.POST['email'],
        password=hashed_password)
    user.save()
    messages.add_message(request, messages.INFO,'User successfully added')
    return redirect(saveUser)

def saveCitizen(request):
    if request.method == 'POST':
        citizen=Person.objects.filter(national_id=request.POST["national_id"])
        if citizen.exists():
            messages.error(request,"Citizen with that National ID already exists")
            return redirect(addCitizen)
        else:
            myfile = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.url(filename)

            person = Person.objects.create(
                name=request.POST["name"],
                national_id=request.POST["national_id"],
                address=request.POST["address"],
                picture=uploaded_file_url[1:],
                status="Free"
            )
            person.save()
            messages.add_message(request, messages.INFO, "Elave Olundu!")
            return redirect(viewCitizens)



def viewCitizens(request):
    citizens=Person.objects.all()
    context={
        "citizens":citizens
    }
    return render(request,'home/view_citizenz.html',context)

def wantedCitizen(request, citizen_id):
    wanted = Person.objects.filter(pk=citizen_id).update(status='Wanted')
    if (wanted):
      
        messages.add_message(request,messages.INFO,"1")
    else:
        messages.error(request,"2")
    return redirect(viewCitizens)

def freeCitizen(request, citizen_id):
    free = Person.objects.filter(pk=citizen_id).update(status='Free')
    if (free):
        messages.add_message(request,messages.INFO,"5")
    else:
        messages.error(request,"6")
    return redirect(viewCitizens)

def spottedCriminals(request):
    thiefs=ThiefLocation.objects.filter(status="Wanted")
    count = ThiefLocation.objects.count()
    context={
        'thiefs':thiefs,
        'count':count
    }
    return render(request,'home/spotted_thiefs.html',context)

def foundThief(request,thief_id):
    free = ThiefLocation.objects.filter(pk=thief_id).order_by('-id')
    freectzn = ThiefLocation.objects.filter(national_id=free.get().national_id).update(status='Found')
    if(freectzn):
        thief = ThiefLocation.objects.filter(pk=thief_id).order_by('-id')
        free = Person.objects.filter(national_id=thief.get().national_id).update(status='Found')
        if(free):
            messages.add_message(request,messages.INFO,"7")
        else:
            messages.error(request,"8")
    return redirect(spottedCriminals)


def viewThiefLocation(request,thief_id):
    thief = ThiefLocation.objects.filter(pk=thief_id).order_by('-id')
    context={
        "thief":thief
    }
    return render(request,'home/loc.html',context)

def viewReports(request):
    return render(request,"home/reports.html")




def login(request):
    pass


def success(request):
    user = User.objects.all()
    context = {
        "user": user
    }
    return render(request, 'home/welcome.html', context)

def detectImage(request):
  
    if request.method == 'POST' and request.FILES['image']:
        myfile = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
   

    images=[]
    encodings=[]
    names=[]
    files=[]

    prsn=Person.objects.all()
    for crime in prsn:
        images.append(crime.name+'_image')
        encodings.append(crime.name+'_face_encoding')
        files.append(crime.picture)
        names.append(crime.name+ ' '+ crime.address)


    for i in range(0,len(images)):
        images[i]=face_recognition.load_image_file(files[i])
        encodings[i]=face_recognition.face_encodings(images[i])[0]






    known_face_encodings = encodings
    known_face_names = names

    unknown_image = face_recognition.load_image_file(uploaded_file_url[1:])

    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

   
    pil_image = Image.fromarray(unknown_image)
    draw = ImageDraw.Draw(pil_image)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

     
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]


        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    del draw

    pil_image.show()
    return redirect('axtar')

def detectWithWebcam(request):
    video_capture = cv2.VideoCapture(0)

    images=[]
    encodings=[]
    names=[]
    files=[]
    nationalIds=[]

    prsn=Person.objects.all()
    for crime in prsn:
        images.append(crime.name+'_image')
        encodings.append(crime.name+'_face_encoding')
        files.append(crime.picture)
        names.append(crime.name)
        nationalIds.append(crime.national_id)


    for i in range(0,len(images)):
        images[i]=face_recognition.load_image_file(files[i])
        encodings[i]=face_recognition.face_encodings(images[i])[0]

    known_face_encodings = encodings
    known_face_names = names
    n_id=nationalIds



    while True:
        ret, frame = video_capture.read()

        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"


            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                ntnl_id = n_id[best_match_index]
                person = Person.objects.filter(national_id=ntnl_id)
                name = known_face_names[best_match_index]


             
                if(person.get().status=='Wanted'):
                    
                    thief = ThiefLocation.objects.create(
                        name=person.get().name,
                        national_id=person.get().national_id,
                        address=person.get().address,
                        picture=person.get().picture,
                        status='Wanted',
                        latitude='20202020',
                        longitude='040404040',
                    )
                   

                    thief.save()



            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return redirect('/success')




