from SocketCommunication.tcp_socket import LocalTCPSocket


class ReplicatedData:
    """
    Inherited from by objects that require to be replicated to UE4
    When replicating, 3 methods are called from replicate_data
        get_data_to_replicate is implemented by the object, returning data of the type required
        prepare_replicated_data is implemented by children for different data types, converting the previous data to dict
        send_data tells the socket to send the dict
    """
    def __init__(self, replicate_type: str, object_name):
        self.replicate_type = replicate_type
        self.object_name = object_name

    def replicate_data(self, socket: LocalTCPSocket):
        data = self.get_data_to_replicate()
        prepared = self.prepare_replicated_data(data)
        self.send_data(socket, prepared)

    def prepare_replicated_data(self, data) -> dict:
        raise NotImplementedError

    def get_data_to_replicate(self):
        raise NotImplementedError

    def send_data(self, socket: LocalTCPSocket, replicate_data):
        socket.send_ue_data(self.replicate_type, self.object_name, replicate_data)


class ReplicatedTransform(ReplicatedData):
    def transform_to_dict(self, loc, rot, scale):
        return {
            "locx": loc[0], "locy": loc[1],
            "rot": rot[0],
            "scalex": scale[0], "scaley": scale[1]
                }

    def prepare_replicated_data(self, data) -> dict:
        # for transform, data must be tuple of loc, rot, scale
        return self.transform_to_dict(data[0], data[1], data[2])

    def get_data_to_replicate(self):
        # implemented by child
        raise NotImplementedError


class ReplicatedScalar(ReplicatedData):
    def scalar_to_dict(self, scalar):
        return {"scalar": scalar}

    def prepare_replicated_data(self, data) -> dict:
        # for scalar, data must be scalar
        return self.scalar_to_dict(data)

    def get_data_to_replicate(self):
        # implemented by child
        raise NotImplementedError
