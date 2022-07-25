"""
TCP Socket used to communicate with Unreal Engine
"""

import socket


class LocalTCPSocket:
    def __init__(self, port):
        self.port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind(("localhost", port))
        self.s.listen(1)

    def send_ue_data(self, data_type: str, object_name: str, data: dict):
        # TYPE;NAME;name1:data1,name2:data2

        #  e.g. CAR;CAR_5;locx:50,locy:50,locz:50

        string = data_type + ";" + object_name + ";" + ",".join([str(entry[0])+":"+str(entry[1]) for entry in data.items()])
        print("SENDING TO UE:", string)
        self.s.sendall(string.encode())
