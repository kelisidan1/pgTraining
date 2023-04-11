# 仅仅作为Connection的集合对象，提供一些集合操作。

class Connections(object):
    def __init__(self):
        self.connections = []
    def add_connections(self,connection):
        self.connections.append(connection)
    def dump(self):
        for conn in self.connections:
            print(conn)

    #LAST TIME

