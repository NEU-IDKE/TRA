from shapely.geometry import Point, LineString, Polygon
import pandas as pd
from multiprocessing import Process, Queue

QSIZE = 60

def process_point_str_list(x):
    a, b = x.strip('() \n').split(' ')
    return float(a), float(b)

def get_object(wkt):
    start = wkt.find('(')
    modal = wkt[0: start].strip().lower()
    point_str = wkt[start:]

    if modal == "point":
        x, y = point_str.strip('() \n').split(' ')
        x, y = float(x), float(y)
        point_2D = Point(x, y)

        return point_2D, 1

    elif modal == "linestring":
        point_str_list = point_str.strip('() \n').split(',')
        points = list(map(process_point_str_list, point_str_list))
        line_2D = LineString(points)
        return line_2D, 2

    elif modal == "polygon":
        point_str_list = point_str.strip('() \n').split(',')
        points = list(map(process_point_str_list, point_str_list))
        polygon_2D = Polygon(points)
        return polygon_2D, 3
    
def calture_topological_relation(ob1, ob2, name1, name2, write=True, path='./topological_relation.txt'):
    object1, object2 = ob1, ob2
    f = open(path, 'a', encoding='utf-8')
    if object1 == object2:
        f.write(name1 + '\tequals\t' + name2 + '\n')

    if object1.overlaps(object2):
        f.write(name1 + '\toverlaps\t' + name2 + '\n')
    elif object1.touches(object2):
        f.write(name1 + '\ttouches\t' + name2 + '\n')
    elif object1.intersects(object2):
        f.write(name1 + '\tintersects\t' + name2 + '\n')


    if object2.within(object1):
        f.write(name1 + '\tcontains\t' + name2 + '\n')
        f.write(name2 + '\twithin\t' + name1 + '\n')
    elif object1.within(object2):
        f.write(name2 + '\tcontains\t' + name1 + '\n')
        f.write(name1 + '\twithin\t' + name2 + '\n')
    
    if object1.covers(object2):
        f.write(name1 + '\tcovers\t' + name2 + '\n')
        f.write(name2 + '\tcoveredBy\t' + name1 + '\n')
    elif object2.covers(object1):
        f.write(name1 + '\tcoveredBy\t' + name2 + '\n')
        f.write(name2 + '\tcovers\t' + name1 + '\n')

    if object1.crosses(object2):
        f.write(name1 + '\tcrosses\t' + name2 + '\n')

    f.close()

class MyThread(Process):
    def __init__(self, ranges, queue=None):
        super().__init__()
        self.jindu = 0
        self.name = ''
        self.queue = queue
        self.kaishi = ranges[0]
        self.jieshu = ranges[1]
        self.wkt = pd.read_csv('../../data/source/GST/spatial_modality.txt', names=['name', 'wkt'], sep='\t')
    
    def run(self):
        wkt = self.wkt
        ranges = [self.kaishi, self.jieshu]
        self.setN(self.kaishi)
        length = wkt.shape[0]
        for i in range(ranges[0], ranges[1]):
            self.setN(i)
            if self.queue:
                if self.queue.full():
                    try:
                        _ = self.queue.get(timeout=10)
                    except Exception as e:
                        print(e)
                self.queue.put([self.getName(), self.getN()])
            for j in range(i + 1, length):
                name1, name2 = wkt['name'].loc[i], wkt['name'].loc[j]
                object1, num1 = get_object(wkt['wkt'].loc[i])
                object2, num2 = get_object(wkt['wkt'].loc[j])
                try:
                    calture_topological_relation(object1, object2, name1, name2)
                except:
                    print(self.name + '-error!')
    def setN(self, n):
        self.jindu = n
    
    def getN(self):
        return self.jindu

    def setName(self, x):
        self.name = x

    def getName(self):
        return self.name
    
def meun():
    print(
"""
1: see the success rate
2: see n
3: see task range of every thread
input:""", end='')
    
def list_schedule(q, schedule):
    while not q.empty():
        t = q.get()
        schedule[t[0]].setN(t[1]) 
    for j in schedule:
        i = schedule[j]
        print(i.getName() + f': {(i.getN() - i.kaishi) / (i.jieshu - i.kaishi)}')

def list_n(q, schedule):
    while not q.empty():
        t = q.get()
        schedule[t[0]].setN(t[1]) 
    for j in schedule:
        i = schedule[j]
        print(i.name + f': {i.getN()}')

def list_range(q, schedule):
    while not q.empty():
        t = q.get()
        schedule[t[0]].setN(t[1]) 
    for j in schedule:
        i = schedule[j]
        print(i.name + f': {i.kaishi} - {i.jieshu}')

if __name__ == '__main__':
    thread_num = 20
    wkt = pd.read_csv('../../data/source/GST/spatial_modality.txt', names=['name', 'wkt'], sep='\t')
    num = wkt.shape[0]
    every_process = num // thread_num
    # threads = []
    jobs = []
    queue = Queue(QSIZE)
    for i in range(thread_num - 1):
        job = MyThread([i * every_process, (i + 1) * every_process], queue)
        job.setName('Process:' + str(i))
        job.start()
        # threads.append(pro)
        jobs.append(job)
    i = 19
    job = MyThread([i * every_process, num], queue)
    job.setName('Process:' + str(i))
    # pro = Process(target=job.run)
    job.start()
    # threads.append(pro)
    jobs.append(job)
    schedule = {'Process:' + str(i):jobs[i] for i in range(thread_num)}
    while 1:
        meun()
        i = input()
        if i == '1':
            list_schedule(queue, schedule)
        elif i == '2':
            list_n(queue, schedule)
        elif i == '3':
            list_range(queue, schedule)
        else:
            print('input error!!!')

    