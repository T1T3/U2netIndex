import threading
 
class ThreadWithReturnValue(threading.Thread):
    def __init__(self, *init_args, **init_kwargs):
        threading.Thread.__init__(self, *init_args, **init_kwargs)
        self._return = None
    def run(self):
        self._return = self._target(*self._args, **self._kwargs)
    def join(self):
        threading.Thread.join(self)
        try:
            return self.result
        except Exception:
            return None
        
# 69/1984=0.034778225806451612903
# (6973-5855)/6973= 0.16033271188871360964



# 8/231*69=2.3896103896103896104
# 8/231*(6973-5855)=38.718614718614718615
