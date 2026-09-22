from advanced_queue_estimator import QueueEstimator

import time

queue = QueueEstimator()



frame1 = [

    {"track_id":1,"center":(100,200)},

    {"track_id":2,"center":(250,220)},

    {"track_id":3,"center":(420,250)}

]

print("Frame 1")

print(queue.compute(frame1))

time.sleep(1)


frame2 = [

    {"track_id":1,"center":(102,201)},  

    {"track_id":2,"center":(251,221)}, 

    {"track_id":3,"center":(480,260)}   
]

print()

print("Frame 2")

result = queue.compute(frame2)

print(result)

print()

print("Queue Length :", result["queue_length"])

print("Average Speed :", result["average_speed"])