"""
TCP Socket used to communicate with Unreal Engine
"""
import global_settings as gs
import time
import socket


class LocalTCPSocket:
    def __init__(self, port):
        self.port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind(("localhost", port))
        print("Waiting for connection")
        self.s.listen(1)
        conn, address = self.s.accept()
        self.conn = conn
        print("Connected")
        self.last_message = time.time()
        self.time_limit = 0.02

    def send_ue_data(self, object_type: str, object_name: str, data_type: str, data: dict):
        # TYPE;NAME;TYPE;name1:data1,name2:data2

        # time limit to stop too many messages
        if time.time() - self.last_message < self.time_limit:
            time.sleep(self.time_limit)

        #  e.g. CAR;CAR_5;LOC;locx:50,locy:50,locz:50
        data_str = ",".join([str(entry[0])+":"+str(entry[1]) for entry in data.items()])
        string = ";".join((object_type, object_name, data_type, data_str))
        self.last_message = time.time()
        padded = string + "@"*(gs.MESSAGE_LENGTH - len(string))
        print("SENDING TO UE:", padded)
        self.conn.sendall(padded.encode())
