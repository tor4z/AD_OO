import threading


class Singleton(object):
    _SINGLETON_LOCK = threading.Lock()
    _INSTABCE = {}

    @classmethod
    def _create_instance(cls, *args, **kwds):
        return super(Singleton, cls).__new__(cls, *args, **kwds)

    def __new__(cls, *args, **kwds):
        if cls not in cls._INSTABCE:
            with cls._SINGLETON_LOCK:
                if cls not in cls._INSTABCE:
                    obj = cls._create_instance(*args, **kwds)
                    cls._INSTABCE[cls] = obj
        return cls._INSTABCE[cls]
