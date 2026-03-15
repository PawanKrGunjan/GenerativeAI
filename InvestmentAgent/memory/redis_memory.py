import redis
import json

r = redis.Redis()

def save_state(thread_id, state):

    r.set(thread_id, json.dumps(state))


def load_state(thread_id):

    data = r.get(thread_id)

    if data:
        return json.loads(data)

    return None