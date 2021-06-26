import os
from cv2 import cv2 
import time

emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
face_cascade = cv2.CascadeClassifier('.models/haarcascade_frontalface_default.xml')

framepath = './dataset/framesalt2/'
videopath = './dataset/train_videos/'
framenum = 0

if not os.path.exists(os.path.join(framepath)):
    os.mkdir(framepath)

#tüm video isimlerini alıyoruz
video_list = os.listdir(videopath)

for i in range(len(video_list)):
    print(f"İşlem: {i+1}/{len(video_list)} ", end="")
    tic = time.perf_counter()
    #videonun üçüncü argümanı duygu durumunu gösteriyor.
    emo = int(video_list[i].split("-")[2]) - 1
    
    #duygular için klasör oluşturuyoruz.
    if not os.path.exists(os.path.join(framepath, emotions[emo])):
        os.mkdir(framepath + emotions[emo])
       
    #videoyu okuyoruz
    videocap = cv2.VideoCapture(os.path.join(*(videopath, video_list[i])))
    success,frame = videocap.read() 
    
    count = 0
    while success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # yüz algılama
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        for (x, y, w, h) in faces:
            sub_face = frame[y:y+h, x:x+w]
        
            #okunan videonun framelerini ait olduğu sınıfın klasörüne kaydediyoruz.
            facespath = os.path.join(*(framepath, emotions[emo])) + "/frame_" + str(framenum) + ".png"
            cv2.imwrite(facespath, sub_face)
            success,frame = videocap.read() 
            framenum += 1
            count += 1
    
    toc = time.perf_counter()
    print(f"{toc - tic:0.2f} saniye sürdü.")