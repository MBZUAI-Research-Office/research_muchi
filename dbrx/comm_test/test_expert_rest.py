import flask
import numpy as np

app = flask.Flask(__name__)
# EXPERT_ID = None


@app.post("/execute")
def execute():
    input = np.frombuffer(flask.request.data, dtype=np.float16)
    resp = flask.make_response(input.tobytes())
    resp.headers.set("Content-Type", "application/octet-stream")
    return resp
    # x = np.array(args["data"])
    # weights = np.random.uniform(-1, 1, size=x.shape)
    # output = np.matmul(x, weights).tolist()
    # return {"status": "success", "output": output}


# @app.post("/config")
# def config():
#     configs = request.get_json()
#     EXPERT_ID = configs["expert_id"]
#     return {"status": f"successfully set expert {EXPERT_ID}"}
