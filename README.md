# LocateNet



1) completed the portal information frr 

2) facial recognition !!

video/image
↓
face detection
↓
face alignment
↓
face enhancement (optional)
↓
face embedding model
↓
vector database search
↓
identity match



Video / Image
↓
YOLOv8 (person detection)
↓
DeepSORT (tracking)
↓
RetinaFace (face detection)
↓
CodeFormer (optional enhancement)
↓
ArcFace / AdaFace (face embeddings)
↓
OSNet (body embeddings)
↓
FAISS vector search
↓
candidate ranking


example 


Database entry

Name: Rahul Sharma
Last seen: Delhi Metro
Photo: passport image

-> two embeddings saved (face,body)


step 1)

input image---> system process it ( face vector ,body vector )

RetinaFace → detect face
↓
ArcFace → generate face embedding
↓
YOLO → detect person body
↓
OSNet → generate body embedding
↓
Store embeddings in FAISS database



step 2) video 

YOLOv8 → detect person
↓
DeepSORT → track person
↓
RetinaFace → try detect face



face visible ?? yes thena arcface embedding then faiss search

face not visible ?? yes then body crop then osnet reid model body embeding faiss search 



if video 

diff frame then avg it up

final scrore = face + body both



Repeated sighting detected

Possible unidentified person

Reports: 4
Locations:
- Delhi metro
- Connaught place
- Rajiv chowk

Similarity score: 0.83


